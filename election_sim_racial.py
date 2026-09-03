import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import choice
from functools import partial
from scipy.stats import t as t_dist
from scipy.spatial import cKDTree

from gerrychain import Graph, Partition, MarkovChain, constraints, updaters
from gerrychain.updaters import cut_edges
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain.tree import recursive_tree_part, bipartition_tree


# ── Geometry / data prep ─────────────────────────────────────────────────────

def _repair_geometries(gdf, label=""):
    invalid = ~gdf.geometry.is_valid
    if not invalid.any():
        return gdf
    from shapely import make_valid
    from shapely.validation import explain_validity

    print(f"repairing {invalid.sum()} invalid geometrie(s) in {label}:")
    for idx in gdf.index[invalid]:
        print(f"  row {idx}: {explain_validity(gdf.geometry[idx])}")

    orig_area = gdf.loc[invalid].geometry.area
    gdf = gdf.copy()
    gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)

    new_area = gdf.loc[invalid].geometry.area
    rel_change = ((new_area - orig_area).abs() /
                  orig_area.replace(0, float("nan"))).fillna(0)
    if (rel_change > 0.001).any():
        print("  WARNING: repair changed area >0.1% for rows "
              f"{list(rel_change[rel_change > 0.001].index)}")
    assert gdf.geometry.is_valid.all(), f"repair failed for {label}"
    return gdf


# PL 94-171 summary-file field codes -> readable column names. Extend per state if a
# state's PL file uses different field codes (rare, but check before reusing as-is).
DEMO_COLS = {
    "P0020001": "TOTPOP",    # total population
    "P0020002": "HISP",      # Hispanic or Latino (any race)
    "P0020005": "NH_WHITE",  # non-Hispanic White alone
    "P0020006": "NH_BLACK",  # non-Hispanic Black alone
    "P0020008": "NH_ASIAN",  # non-Hispanic Asian alone
    "P0040001": "VAP",       # total voting-age population
    "P0040002": "HVAP",      # Hispanic VAP
    "P0040005": "NH_WVAP",   # non-Hispanic White VAP
    "P0040006": "NH_BVAP",   # non-Hispanic Black VAP
    "P0040007": "NH_NVAP",   # non-Hispanic Native American VAP
    "P0040008": "NH_AVAP",   # non-Hispanic Asian VAP
}

# race_list key -> the VAP column in DEMO_COLS it reads from. Extend to match race_list.
RACE_VAP_COL = {
    "white":  "NH_WVAP",
    "asian":  "NH_AVAP",
    "latino": "HVAP",
    "black":  "NH_BVAP",
    "native": "NH_NVAP",
}


def data_merge(block_shp, prec_shp, comb_summary, state, crs):
    # ---- 1. Load geometries, repair, then force a common CRS -----------
    gdfprec = _repair_geometries(gpd.read_file(prec_shp), label="precincts").to_crs(crs)
    gdfblock = _repair_geometries(gpd.read_file(block_shp), label="blocks").to_crs(crs)

    # ---- 2. Block-level demographics from the PL file ------------------
    pl = pd.read_csv(comb_summary, header=0, encoding="latin-1",
                      on_bad_lines="skip", dtype={"GEOID": str})
    plblock = pl[pl["SUMLEV"] == 750].copy()  # 750 = block-level rows
    plblock["BLOCK_GEOID"] = plblock["GEOID"].str[-15:]
    plblock = (plblock[["BLOCK_GEOID"] + list(DEMO_COLS)]
               .rename(columns=DEMO_COLS))

    gdfblock = gdfblock.merge(plblock, left_on="GEOID20",
                               right_on="BLOCK_GEOID", how="left")
    n_missing = gdfblock["TOTPOP"].isna().sum()
    if n_missing:
        print(f"warning: {n_missing} blocks had no PL match; filling 0")
    gdfblock[list(DEMO_COLS.values())] = (
        gdfblock[list(DEMO_COLS.values())].fillna(0)
    )

    # ---- 3. Assign each block to a precinct -----------------------------
    pts = gdfblock[["GEOID20", "geometry"] + list(DEMO_COLS.values())].copy()
    pts["geometry"] = pts.representative_point()
    joined = gpd.sjoin(pts, gdfprec[["UNIQUE_ID", "geometry"]],
                        how="left", predicate="within")

    unassigned = joined["UNIQUE_ID"].isna()
    lost_pop = joined.loc[unassigned, "TOTPOP"].sum()
    print(f"blocks unassigned to any precinct: {unassigned.sum()} "
          f"(population {lost_pop:.0f})")

    # ---- 4. Aggregate blocks -> precincts (exact sum, blocks nest) -----
    demo = (joined.dropna(subset=["UNIQUE_ID"])
                  .groupby("UNIQUE_ID")[list(DEMO_COLS.values())]
                  .sum()
                  .reset_index())

    # ---- 5. Merge demographics onto the precinct election file ---------
    gdfprec_merged = gdfprec.merge(demo, on="UNIQUE_ID", how="left")
    empty = gdfprec_merged["TOTPOP"].isna()
    if empty.sum():
        print(f"warning: {empty.sum()} precincts received no blocks "
              f"(check IDs): {gdfprec_merged.loc[empty, 'UNIQUE_ID'].tolist()[:10]}")
        gdfprec_merged[list(DEMO_COLS.values())] = (
            gdfprec_merged[list(DEMO_COLS.values())].fillna(0)
        )

    # ---- 6. Sanity check: no people lost in the pipeline ---------------
    block_total = gdfblock["TOTPOP"].sum()
    prec_total = gdfprec_merged["TOTPOP"].sum()
    print(f"state total from blocks:    {block_total:,.0f}")
    print(f"state total on precincts:   {prec_total:,.0f}")
    print(f"difference:                 {block_total - prec_total:,.0f}")

    # ---- 7. Convenience columns for the analysis -----------------------
    gdfprec_merged["POC_VAP"] = (
        gdfprec_merged["VAP"] - gdfprec_merged["NH_WVAP"]
    )
    for col in ["HVAP", "NH_BVAP", "NH_AVAP", "NH_WVAP", "NH_NVAP", "POC_VAP"]:
        share = gdfprec_merged[col] / gdfprec_merged["VAP"].replace(0, pd.NA)
        gdfprec_merged[col + "_PCT"] = share.fillna(0)

    # ---- 8. Build and cache the GerryChain graph ------------------------
    graph = Graph.from_geodataframe(gdfprec_merged)
    graph.to_json(f"{state}_precincts.json")
    print(f"graph saved to {state}_precincts.json "
          f"({len(graph.nodes)} nodes)")

    return gdfprec_merged


def connect_islands(graph):
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) <= 1:
        return graph
    main_ids = list(components[0])
    main_coords = np.array([[graph.geometry[i].centroid.x, graph.geometry[i].centroid.y] for i in main_ids])
    tree = cKDTree(main_coords)
    for comp in components[1:]:
        comp_ids = list(comp)
        comp_coords = np.array([[graph.geometry[i].centroid.x, graph.geometry[i].centroid.y] for i in comp_ids])
        dist, idx = tree.query(comp_coords)
        best = np.argmin(dist)
        graph.add_edge(comp_ids[best], main_ids[idx[best]])
    return graph


def build_chain(pop_col, gdfprec_merged, num_districts, has_islands=False, epsilon=0.05, node_repeats=2):
    """epsilon/node_repeats control population-balance tolerance and spanning-tree redraws
    for the ReCom chain itself. Default (0.05, 2) matches every state except NY's SMD/
    plurality chain, which needs a much looser epsilon=0.5, node_repeats=20 to successfully
    partition into 63 districts -- pass those explicitly for that call only."""
    graphprec = Graph.from_geodataframe(gdfprec_merged)
    if has_islands:
        graphprec = connect_islands(graphprec)

    myupdaters = {
        "population": updaters.Tally(pop_col, alias="population"),
        "cut_edges": cut_edges,
    }
    total_population = sum([graphprec.nodes[n][pop_col] for n in graphprec.nodes])
    pop_target = total_population / num_districts

    if has_islands:
        assignment = None
        for seed in range(20):
            random.seed(seed)
            try:
                assignment = recursive_tree_part(
                    graphprec, range(1, num_districts + 1),
                    pop_target, pop_col, epsilon,
                    node_repeats=10,
                    method=partial(bipartition_tree, max_attempts=50000),
                )
                break
            except RuntimeError:
                continue
        if assignment is None:
            raise RuntimeError(
                f"recursive_tree_part failed to partition into {num_districts} districts "
                "after 20 random-seed retries at max_attempts=50000."
            )
    else:
        assignment = recursive_tree_part(
            graphprec, range(1, num_districts + 1),
            pop_target, pop_col, epsilon
        )

    initial_partition = Partition(graphprec, assignment, myupdaters)
    myconstraints = [constraints.within_percent_of_ideal_population(initial_partition, epsilon)]
    myproposal = partial(
        recom, pop_col=pop_col,
        pop_target=pop_target,
        epsilon=epsilon, node_repeats=node_repeats
    )
    chain = MarkovChain(
        proposal=myproposal, constraints=myconstraints,
        accept=always_accept, initial_state=initial_partition, total_steps=10000
    )
    selected_indices = set(random.sample(range(1000, 10000), 100))
    sampled_maps = []
    for i, partition in enumerate(chain):
        if i in selected_indices:
            sampled_maps.append(partition)
    return sampled_maps


# ── Ballot generation ─────────────────────────────────────────────────────────
# NOTE: these loop over an arbitrary-length race_list (any number of groups), unlike the
# partisan file's 2-party-only versions. Do not swap between the two files.

def luce(num_ballots, mean_support_by_race, cand_list, vote_portion_by_race, race_list):
    ballots_list = []
    for race in vote_portion_by_race.keys():
        num_ballots_race = int(num_ballots * vote_portion_by_race[race])
        cand_support_vec = list(mean_support_by_race[race].values())
        total = sum(cand_support_vec)
        cand_support_vec = [p / total for p in cand_support_vec]
        for j in range(num_ballots_race):
            ballot = list(choice(cand_list, len(cand_list), p=cand_support_vec, replace=False))
            ballots_list.append(ballot)
    return ballots_list


def plu(num_ballots, mean_support_by_race, cand_list, vote_portion_by_race, race_list):
    ballots_list = []
    for race in vote_portion_by_race.keys():
        num_ballots_race = int(num_ballots * vote_portion_by_race[race])
        cand_support_vec = list(mean_support_by_race[race].values())
        total = sum(cand_support_vec)
        cand_support_vec = [p / total for p in cand_support_vec]
        for j in range(num_ballots_race):
            ballot = list(choice(cand_list, size=1, p=cand_support_vec, replace=False))
            ballots_list.append(ballot)
    return ballots_list


# ── Counting ────────────────────────────────────────────────────────────────

def recompute_count(candidates, ballot_list):
    cand_totals = {}
    for cand in candidates:
        cand_totals[cand] = len([ballot for ballot in ballot_list if ballot[0] == cand])
    return cand_totals


def remove_cand(cand, ballot_list):
    for n, ballot in enumerate(ballot_list):
        new_ballot = []
        for c in ballot:
            if c != cand:
                new_ballot.append(c)
        ballot_list[n] = new_ballot


def cincinnati_transfer(cand, ballot_list, win_lose, cutoff):
    if win_lose == 'lose':
        remove_cand(cand, ballot_list)
    else:
        cand_ballots_index = []
        single_cand_ballots_index = []
        for n, ballot in enumerate(ballot_list):
            if ballot[0] == cand and len(ballot) == 1:
                single_cand_ballots_index.append(n)
            elif ballot[0] == cand and len(ballot) > 1:
                cand_ballots_index.append(n)
        rand_winners1 = random.sample(single_cand_ballots_index, min(int(cutoff), len(single_cand_ballots_index)))
        rand_winners2 = random.sample(cand_ballots_index, int(cutoff) - len(rand_winners1))
        rand_winners = rand_winners1 + rand_winners2
        for index in sorted(rand_winners, reverse=True):
            del ballot_list[index]
        remove_cand(cand, ballot_list)


def rcv_run(ballot_list, cand_list, num_seats, transfer_method, verbose_bool=False):
    winners = []
    cutoff = int(len(ballot_list) / (num_seats + 1) + 1)
    candidates = cand_list.copy()
    ballot_list = [x for x in ballot_list if x != []]
    cand_totals = recompute_count(candidates, ballot_list)
    while len(winners) < num_seats:
        remaining_cands = candidates
        if len(remaining_cands) == num_seats - len(winners):
            winners = winners + remaining_cands
            break
        cand_totals = recompute_count(candidates, ballot_list)
        for cand in list(candidates):
            if len(winners) == num_seats:
                break
            if cand_totals[cand] >= cutoff:
                winners.append(cand)
                transfer_method(cand, ballot_list, "win", cutoff)
                candidates.remove(cand)
                ballot_list = [x for x in ballot_list if x != []]
                cand_totals = recompute_count(candidates, ballot_list)
                if verbose_bool:
                    print("candidate", cand, "elected")

        if len(winners) == num_seats:
            break
        min_count = min(cand_totals.values())
        min_cand_list = [k for k, v in cand_totals.items() if v == min_count]
        min_cand = random.choice(min_cand_list)
        transfer_method(min_cand, ballot_list, "lose", cutoff)
        candidates.remove(min_cand)
        ballot_list = [x for x in ballot_list if x != []]
        cand_totals = recompute_count(candidates, ballot_list)
        if verbose_bool:
            print("candidate", min_cand, "eliminated")
    return winners


def swc_run(ballot_list, cand_list):
    vote_counts = recompute_count(cand_list, ballot_list)
    winner = max(vote_counts, key=vote_counts.get)
    return winner


def largest_remainder(props, total):
    quotas = {r: props[r] * total for r in props}
    floors = {r: int(q) for r, q in quotas.items()}
    remainders = {r: quotas[r] - floors[r] for r in quotas}
    deficit = total - sum(floors.values())
    for r in sorted(remainders, key=remainders.get, reverse=True)[:deficit]:
        floors[r] += 1
    return floors


# ── Competitiveness ───────────────────────────────────────────────────────────

def compute_competitiveness(df, share_col, group_col="district", sigma=None):
    df = df.copy()
    if sigma is None:
        sigma_by_group = df.groupby(group_col)[share_col].std()
        df["sigma"] = df[group_col].map(sigma_by_group)
    else:
        df["sigma"] = sigma
    df["M"] = df[share_col] - 0.5
    df["pk"] = t_dist.cdf(df["M"] / df["sigma"], df=2)
    df["C"] = np.minimum(df["pk"], 1 - df["pk"])
    return df


# ── Top-level simulations ──────────────────────────────────────────────────────

def election_sim(sampled_maps, num_districts, num_seats, num_cands, cand_list, race_list, pi, sigma=None):
    # ── 1. Per-district racial VAP proportions ────────────────────────────
    pop_rows = []
    for i, step in enumerate(sampled_maps):
        for d in step.parts:
            vap = sum(step.graph.nodes[n]["VAP"] for n in step.parts[d])
            if vap == 0:
                continue
            row = {"step": i, "district": d}
            for r, col in RACE_VAP_COL.items():
                row[f"{r}_prop"] = sum(step.graph.nodes[n][col] for n in step.parts[d]) / vap
            pop_rows.append(row)
    df_pop = pd.DataFrame(pop_rows)

    # ── 2. Candidate count assignment (largest remainder) ─────────────────
    cand_rows = []
    for i in df_pop["step"].unique():
        entry = {"step": i}
        for d in df_pop["district"].unique():
            row = df_pop[(df_pop["step"] == i) & (df_pop["district"] == d)].iloc[0]
            props = {r: row[f"{r}_prop"] for r in race_list}
            counts = largest_remainder(props, num_cands)
            for r in race_list:
                entry[f"n_{r}_d{d}"] = counts[r]
        cand_rows.append(entry)
    df_cand = pd.DataFrame(cand_rows)

    # ── 3. Vote portions, support probabilities, candidate-race mapping ───
    vote_portion_list, mean_support_list, race_cand_rows = [], [], []
    cands_by_race_all = {}   # {step: {district: {race: [candidates]}}}

    for step in df_cand["step"].unique():
        step_info = df_cand[df_cand["step"] == step].iloc[0]
        pop_data = df_pop[df_pop["step"] == step].set_index("district")

        vote_entry, support_entry = {"step": step}, {"step": step}
        race_cands = {r: [] for r in race_list}
        cands_by_race_all[step] = {}

        for d in range(1, num_districts + 1):
            vote_entry[f"d{d}"] = {r: pop_data.loc[d, f"{r}_prop"] for r in race_list}

            block = cand_list[(d - 1) * num_cands: d * num_cands]
            idx, cands_by_race = 0, {}
            for r in race_list:
                n_r = step_info[f"n_{r}_d{d}"]
                cands_by_race[r] = block[idx: idx + n_r]
                idx += n_r
                race_cands[r].extend(cands_by_race[r])

            # Support probabilities: pi[r] on own-race, 1-pi[r] spread over others
            all_cands = block
            d_supp = {}
            for voter_r in race_list:
                my = cands_by_race[voter_r]
                other = [c for c in all_cands if c not in my]
                p_in = pi[voter_r] / len(my) if my else 0
                p_out = (1 - pi[voter_r]) / len(other) if other else 0
                d_supp[voter_r] = {
                    c: (p_in if c in my else p_out) for c in all_cands
                }
            support_entry[f"d{d}"] = d_supp
            cands_by_race_all[step][d] = cands_by_race

        vote_portion_list.append(vote_entry)
        mean_support_list.append(support_entry)
        race_cand_rows.append(race_cands)

    df_vote_portion = pd.DataFrame(vote_portion_list)
    df_mean_support = pd.DataFrame(mean_support_list)
    df_race_cands = pd.DataFrame(race_cand_rows)

    # ── 4. Generate ballots ───────────────────────────────────────────────
    ballots = []
    for i, step in enumerate(sampled_maps):
        sup = df_mean_support[df_mean_support["step"] == i].iloc[0]
        por = df_vote_portion[df_vote_portion["step"] == i].iloc[0]
        entry = {"step": i}
        for d in range(1, num_districts + 1):
            block = cand_list[(d - 1) * num_cands: d * num_cands]
            entry[f"ballots_d{d}"] = luce(10000, sup[f"d{d}"], block, por[f"d{d}"], race_list)
        ballots.append(entry)
    bl = pd.DataFrame(ballots)

    # ── 5. Run RCV ────────────────────────────────────────────────────────
    winners = []
    white_share_rows = []
    for i, step in enumerate(sampled_maps):
        step_data = bl[bl["step"] == i]
        result = {"step": i}
        for d in range(1, num_districts + 1):
            ballots_d = step_data[f"ballots_d{d}"].values[0]
            result[f"wn_d{d}"] = rcv_run(
                ballots_d,
                cand_list[(d - 1) * num_cands: d * num_cands],
                num_seats,
                cincinnati_transfer,
                verbose_bool=False,
            )

            # racial competitiveness = share of ballots ranking a white candidate first
            white_cands = set(cands_by_race_all[i][d].get("white", []))
            first_choices = [ballot[0] for ballot in ballots_d if ballot]
            white_vote_share = (sum(1 for c in first_choices if c in white_cands)
                                 / len(first_choices)) if first_choices else np.nan
            white_share_rows.append({"step": i, "district": d, "white_vote_share": white_vote_share})
        winners.append(result)
    wn = pd.DataFrame(winners)

    # ── 6. Competitiveness ────────────────────────────────────────────────
    white_share_df = pd.DataFrame(white_share_rows)
    election_margins_df = compute_competitiveness(white_share_df, "white_vote_share", "district", sigma=sigma)

    return wn, df_race_cands, df_vote_portion, election_margins_df


def election_sim_c(sampled_maps_c, num_districts_c, num_cands_c, cand_list, race_list, pi, sigma=None):
    # ── 1. Per-district racial VAP proportions ────────────────────────────
    pop_rows = []
    for i, step in enumerate(sampled_maps_c):
        for d in step.parts:
            vap = sum(step.graph.nodes[n]["VAP"] for n in step.parts[d])
            if vap == 0:
                continue
            row = {"step": i, "district": d}
            for r, col in RACE_VAP_COL.items():
                row[f"{r}_prop"] = sum(step.graph.nodes[n][col] for n in step.parts[d]) / vap
            pop_rows.append(row)
    df_pop = pd.DataFrame(pop_rows)

    # ── 2. Candidate count assignment (largest remainder) ─────────────────
    cand_rows = []
    for i in df_pop["step"].unique():
        entry = {"step": i}
        for d in df_pop["district"].unique():
            row = df_pop[(df_pop["step"] == i) & (df_pop["district"] == d)].iloc[0]
            props = {r: row[f"{r}_prop"] for r in race_list}
            counts = largest_remainder(props, num_cands_c)
            for r in race_list:
                entry[f"n_{r}_d{d}"] = counts[r]
        cand_rows.append(entry)
    df_cand = pd.DataFrame(cand_rows)

    # ── 3. Vote portions, support probabilities, candidate-race mapping ───
    vote_portion_list, mean_support_list, race_cand_rows = [], [], []
    cands_by_race_all = {}

    for step in df_cand["step"].unique():
        step_info = df_cand[df_cand["step"] == step].iloc[0]
        pop_data = df_pop[df_pop["step"] == step].set_index("district")

        vote_entry, support_entry = {"step": step}, {"step": step}
        race_cands = {r: [] for r in race_list}
        cands_by_race_all[step] = {}

        for d in range(1, num_districts_c + 1):
            vote_entry[f"d{d}"] = {r: pop_data.loc[d, f"{r}_prop"] for r in race_list}

            block = cand_list[(d - 1) * num_cands_c: d * num_cands_c]
            idx, cands_by_race = 0, {}
            for r in race_list:
                n_r = step_info[f"n_{r}_d{d}"]
                cands_by_race[r] = block[idx: idx + n_r]
                idx += n_r
                race_cands[r].extend(cands_by_race[r])

            all_cands = block
            d_supp = {}
            for voter_r in race_list:
                my = cands_by_race[voter_r]
                other = [c for c in all_cands if c not in my]
                p_in = pi[voter_r] / len(my) if my else 0
                p_out = (1 - pi[voter_r]) / len(other) if other else 0
                d_supp[voter_r] = {
                    c: (p_in if c in my else p_out) for c in all_cands
                }
            support_entry[f"d{d}"] = d_supp
            cands_by_race_all[step][d] = cands_by_race

        vote_portion_list.append(vote_entry)
        mean_support_list.append(support_entry)
        race_cand_rows.append(race_cands)

    df_vote_portion = pd.DataFrame(vote_portion_list)
    df_mean_support = pd.DataFrame(mean_support_list)
    df_race_cands = pd.DataFrame(race_cand_rows)

    # ── 4. Generate ballots ───────────────────────────────────────────────
    ballots = []
    for i, step in enumerate(sampled_maps_c):
        sup = df_mean_support[df_mean_support["step"] == i].iloc[0]
        por = df_vote_portion[df_vote_portion["step"] == i].iloc[0]
        entry = {"step": i}
        for d in range(1, num_districts_c + 1):
            block = cand_list[(d - 1) * num_cands_c: d * num_cands_c]
            entry[f"ballots_d{d}"] = plu(10000, sup[f"d{d}"], block, por[f"d{d}"], race_list)
        ballots.append(entry)
    bl = pd.DataFrame(ballots)

    # ── 5. Run SWC (plurality) ─────────────────────────────────────────────
    winners = []
    white_share_rows = []
    for i, step in enumerate(sampled_maps_c):
        step_data = bl[bl["step"] == i]
        result = {"step": i}
        for d in range(1, num_districts_c + 1):
            ballots_d = step_data[f"ballots_d{d}"].values[0]
            result[f"wn_d{d}"] = swc_run(
                ballots_d,
                cand_list[(d - 1) * num_cands_c: d * num_cands_c],
            )

            # racial competitiveness = share of ballots ranking a white candidate first
            white_cands = set(cands_by_race_all[i][d].get("white", []))
            first_choices = [ballot[0] for ballot in ballots_d if ballot]
            white_vote_share = (sum(1 for c in first_choices if c in white_cands)
                                 / len(first_choices)) if first_choices else np.nan
            white_share_rows.append({"step": i, "district": d, "white_vote_share": white_vote_share})
        winners.append(result)
    wn = pd.DataFrame(winners)

    # ── 6. Competitiveness ────────────────────────────────────────────────
    white_share_df = pd.DataFrame(white_share_rows)
    election_margins_df_c = compute_competitiveness(white_share_df, "white_vote_share", "district", sigma=sigma)

    return wn, df_race_cands, df_vote_portion, election_margins_df_c


def compute_seat_shares(wn, df_race_cands, num_districts, num_elected, race_list):
    seat_shares = {r: [] for r in race_list}
    for i in wn["step"]:
        step_data = wn[wn["step"] == i]
        winners = sum([list(step_data[f"wn_d{d}"].iloc[0]) for d in range(1, num_districts + 1)], [])
        for r in race_list:
            r_cands = df_race_cands[r].iloc[i]
            r_count = sum(1 for c in r_cands if c in winners)
            seat_shares[r].append(r_count / num_elected)
    return seat_shares


def compute_seat_shares_swc(wn_c, df_race_cands_c, num_districts_c, num_elected, race_list):
    """SMD analog of compute_seat_shares: one winner per district (not a multi-seat list)."""
    seat_shares_swc = {r: [] for r in race_list}
    for i in wn_c["step"]:
        step_data = wn_c[wn_c["step"] == i]
        winners = [step_data[f"wn_d{d}"].iloc[0] for d in range(1, num_districts_c + 1)]
        for r in race_list:
            r_cands = df_race_cands_c[r].iloc[i]
            r_count = sum(1 for c in r_cands if c in winners)
            seat_shares_swc[r].append(r_count / num_elected)
    return seat_shares_swc


# ── Plotting ──────────────────────────────────────────────────────────────────

def compute_state_props(gdfprec_merged, race_list):
    """Statewide VAP proportion per race group, via RACE_VAP_COL's column mapping."""
    total_vap = gdfprec_merged["VAP"].sum()
    return {r: gdfprec_merged[RACE_VAP_COL[r]].sum() / total_vap for r in race_list}


def plot_seat_shares(seat_shares, state_props, status_quo, race_list):
    """Bubble plot of the simulated seat-share distribution per race group. Yellow square
    marks each group's statewide VAP proportion; red dot marks its status-quo seat share."""
    fig, ax = plt.subplots(figsize=(5.5, 8))
    MAX_BUBBLE_AREA = 250

    for i, r in enumerate(race_list):
        y = len(race_list) - 1 - i
        values, counts = np.unique(np.round(seat_shares[r], 2), return_counts=True)
        freqs = counts / counts.max()
        ax.scatter(values * 100, np.full_like(values, y, dtype=float),
                   s=freqs * MAX_BUBBLE_AREA,
                   color="#3D76B4", alpha=0.95, edgecolors="none", zorder=2)
        # Yellow square: statewide VAP proportion
        ax.scatter(state_props[r] * 100, y,
                   s=50, color="yellow", marker="s", edgecolors="black", linewidths=0.5, zorder=3)
        # Red dot: status quo seat share
        ax.scatter(status_quo[r] * 100, y,
                   s=50, color="red", marker="o", zorder=3)

    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(race_list)))
    ax.set_yticklabels(list(reversed(race_list)))
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_competitiveness_violin(election_margins_df, election_margins_df_c, state):
    """Violin plot comparing RCV vs. SMD racial competitiveness (C) distributions for `state`."""
    fig, (ax_rcv, ax_smd) = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True,
        gridspec_kw={"width_ratios": [1, 1]}
    )

    ax_rcv.violinplot([election_margins_df["C"].dropna().values], positions=[1], showmedians=True)
    ax_rcv.set_xticks([1])
    ax_rcv.set_xticklabels(["All districts"])
    ax_rcv.set_ylabel("Competitiveness C = min(pk, 1-pk)")
    ax_rcv.set_title("Multi-seat RCV")
    ax_rcv.set_ylim(0, 0.5)
    ax_rcv.grid(axis="y", linestyle="--", alpha=0.3)

    ax_smd.violinplot([election_margins_df_c["C"].dropna().values], positions=[1], showmedians=True)
    ax_smd.set_xticks([1])
    ax_smd.set_xticklabels(["All districts"])
    ax_smd.set_xlabel("Single-member district")
    ax_smd.set_title("SMD (Plurality)")
    ax_smd.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(f"{state} — Racial Competitiveness (t-distribution score): RCV vs SMD", fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig, (ax_rcv, ax_smd)


def plot_competitiveness_cdf(election_margins_df, election_margins_df_c, state):
    """CDF plot comparing RCV vs. SMD racial competitiveness (C) distributions for `state`."""
    fig, (ax_rcv, ax_smd) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, gridspec_kw={"wspace": 0.15})

    values = election_margins_df["C"].dropna().values
    sorted_vals = np.sort(values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax_rcv.plot(sorted_vals, ecdf, lw=2, color="tab:blue", label="All districts")
    ax_rcv.set_xlabel("Competitiveness C = min(pk, 1-pk)")
    ax_rcv.set_ylabel("Cumulative probability")
    ax_rcv.set_xlim(0, 0.5)
    ax_rcv.set_title(f"{state} — Multi-seat RCV")
    ax_rcv.legend(fontsize=9)
    ax_rcv.grid(linestyle="--", alpha=0.3)

    values = election_margins_df_c["C"].dropna().values
    sorted_vals = np.sort(values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax_smd.plot(sorted_vals, ecdf, lw=2, color="tab:blue", label="All districts")
    ax_smd.set_xlabel("Competitiveness C = min(pk, 1-pk)")
    ax_smd.set_xlim(0, 0.5)
    ax_smd.set_title(f"{state} — SMD (Plurality)")
    ax_smd.legend(fontsize=9)
    ax_smd.grid(linestyle="--", alpha=0.3)

    plt.suptitle("t-distribution Racial Competitiveness CDF: RCV vs SMD", fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig, (ax_rcv, ax_smd)
