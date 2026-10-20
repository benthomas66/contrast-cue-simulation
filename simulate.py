import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- 1. Display: 19 candidate locations (center, six at 4 degrees, twelve at 8; an assumed layout) ----
NP = 19
angles_6 = np.deg2rad(np.arange(0, 360, 60))
angles_12 = np.deg2rad(np.arange(0, 360, 30))
POS = np.zeros((NP, 2))                                           # x, y in degrees
POS[1:7, 0], POS[1:7, 1] = 4 * np.cos(angles_6), 4 * np.sin(angles_6)
POS[7:19, 0], POS[7:19, 1] = 8 * np.cos(angles_12), 8 * np.sin(angles_12)
DIST = np.zeros((NP, NP))                                         # DIST[g, k]: degrees from eye position g to location k
for g in range(NP):
    for k in range(NP):
        dx = POS[g, 0] - POS[k, 0]
        dy = POS[g, 1] - POS[k, 1]
        DIST[g, k] = np.sqrt(dx * dx + dy * dy)
CENTER = 0

# ---- 2. Visibility: weak d' falls off with eccentricity (Zhang & Geisler 2025, Eq. 6); strong is twice weak ----
PEAK = 3.0                      # weak-target d' at fixation
E2 = 4.68                       # eccentricity (degrees) where d' is half its peak
CONTRAST_RATIO = 2.0            # strong d' / weak d'; written m = 2 in notes.md

def dprime_weak(ecc):
    return PEAK * E2 / (ecc + E2)

# ---- 3. One look: every location returns mean + standard-normal noise; mean is 0 off target, d' at the target ----
def observe(target, contrast, noise_row, g):
    mean = np.zeros(NP)
    if target >= 0:
        mean[target] = dprime_weak(DIST[g, target])
        if contrast == 1:       # 0 means weak, 1 means strong
            mean[target] = CONTRAST_RATIO * mean[target]
    return mean + noise_row

# ---- 4. Prior over 39 hypotheses: 19 weak-target locations, 19 strong-target locations, absent ----
P_ABSENT = 0.5

def prior(cue):
    if cue == "W":
        weak_share, strong_share = 1.0, 0.0
    elif cue == "S":
        weak_share, strong_share = 0.0, 1.0
    elif cue == "E":             # Either
        weak_share, strong_share = 0.5, 0.5
    else:
        raise ValueError("The cue must be W, S or E.")
    per_location = (1 - P_ABSENT) / NP                           # 1/38 per location; under Either 1/76 per contrast
    weak_priors = np.full(NP, per_location * weak_share)
    strong_priors = np.full(NP, per_location * strong_share)
    return np.concatenate([weak_priors, strong_priors, [P_ABSENT]])

# ---- 5. Evidence: P = sum of d'^2 and W = sum of d' * R over looks, per location; posterior = prior x likelihood ratio ----
def posterior(P, W, cue):
    ratio_weak = np.exp(W - 0.5 * P)                              # likelihood ratio, weak target at k vs absent
    ratio_strong = np.exp(CONTRAST_RATIO * W - 0.5 * CONTRAST_RATIO ** 2 * P)   # the same with the strong d' map
    weight = prior(cue) * np.concatenate([ratio_weak, ratio_strong, [1.0]])     # absent vs absent is 1
    return weight / weight.sum()

def location_probability(post):                                  # 19 numbers: contrast summed out; absent left aside
    return post[:NP] + post[NP:2 * NP]

# np.argmax chooses the lowest-numbered location if maximum probabilities tie.
def choose_map(P, W):                                             # look at the most probable target location
    return int(np.argmax(location_probability(posterior(P, W, "E"))))

# ---- 6. Trials: all absent, seeded, drawn in the order of the earlier package so its numbers reproduce ----
N_TRIALS = 2000
SEED = 52001
N_OBS = 20
OBSERVATION_COUNTS = (2, 4, 8, 20)                                # read the evidence after each of these numbers of looks

def make_trials():
    rng = np.random.default_rng(SEED)
    trials = []
    for _ in range(N_TRIALS):
        rng.standard_normal(NP)                                   # unused draw, kept for the saved random stream
        noise = rng.standard_normal((N_OBS, NP))
        rng.permutation(NP)                                       # unused draw, kept for the saved random stream
        rng.integers(0, 2 ** 31)                                  # unused draw, kept for the saved random stream
        trials.append(noise)
    return trials

# ---- 7. Search under Either; keep the eye position and the evidence (P, W) after every look ----
def search(trials):
    paths = np.zeros((N_TRIALS, N_OBS), dtype=int)
    P_all = np.zeros((N_TRIALS, N_OBS, NP))
    W_all = np.zeros((N_TRIALS, N_OBS, NP))
    for i in range(N_TRIALS):
        P = np.zeros(NP)
        W = np.zeros(NP)
        for t in range(N_OBS):
            if t == 0:
                g = CENTER
            else:
                g = choose_map(P, W)
            d = dprime_weak(DIST[g])                              # d' of every location from this eye position
            R = observe(-1, 0, trials[i][t], g)                   # -1 means no target anywhere
            P = P + d ** 2
            W = W + d * R
            paths[i, t] = g
            P_all[i, t] = P
            W_all[i, t] = W
    return paths, P_all, W_all

# ---- 8. Re-read the evidence under each label; distance from the best location to the nearest prior fixation ----
def decode(paths, P_all, W_all):
    n_counts = len(OBSERVATION_COUNTS)
    best = {"W": np.zeros((N_TRIALS, n_counts), dtype=int), "E": np.zeros((N_TRIALS, n_counts), dtype=int),
            "S": np.zeros((N_TRIALS, n_counts), dtype=int)}
    distance = {"W": np.zeros((N_TRIALS, n_counts)), "E": np.zeros((N_TRIALS, n_counts)), "S": np.zeros((N_TRIALS, n_counts))}
    p_present_strong = np.zeros((N_TRIALS, n_counts))
    for i in range(N_TRIALS):
        for j in range(n_counts):
            b = OBSERVATION_COUNTS[j]
            P = P_all[i, b - 1]                                    # evidence after b observations
            W = W_all[i, b - 1]
            fixations = paths[i, :b]
            nearest = DIST[fixations].min(axis=0)                  # for each location, degrees to the nearest fixation so far
            for cue in ("W", "E", "S"):
                k = int(np.argmax(location_probability(posterior(P, W, cue))))
                best[cue][i, j] = k
                distance[cue][i, j] = nearest[k]
            p_present_strong[i, j] = location_probability(posterior(P, W, "S")).sum()
    return best, distance, p_present_strong

def main():
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    trials = make_trials()
    paths, P_all, W_all = search(trials)
    best, distance, p_present_strong = decode(paths, P_all, W_all)
    n_counts = len(OBSERVATION_COUNTS)
    mean = np.zeros(n_counts)
    se = np.zeros(n_counts)
    for j in range(n_counts):
        paired_differences = distance["S"][:, j] - distance["W"][:, j]       # one number per trial
        mean[j] = paired_differences.mean()
        se[j] = paired_differences.std(ddof=1) / np.sqrt(N_TRIALS)

    # ---- checks ----
    label_weak = np.concatenate([np.ones(NP), np.zeros(NP), [0.5]])            # likelihood of a late Weak label: 1 / 0 / 1/2
    for i in range(N_TRIALS):
        for j in range(n_counts):
            b = OBSERVATION_COUNTS[j]
            P = P_all[i, b - 1]
            W = W_all[i, b - 1]
            either = posterior(P, W, "E")
            assert abs(either.sum() - 1.0) < 1e-12                             # posterior normalized
            updated = either * label_weak
            assert np.allclose(updated / updated.sum(), posterior(P, W, "W"), atol=1e-12, rtol=0)   # late label = Bayes update
            assert P[best["S"][i, j]] <= P[best["W"][i, j]] + 1e-10            # Strong winner never has more precision
    reference_mean = (1.5512317140714964, 2.1709300786023324, 2.2965981631285084, 2.375417597712765)
    reference_se = (0.057137827904888985, 0.06545694547771021, 0.06233458766849001, 0.047111779274809466)
    assert np.array_equal(mean, reference_mean), "means differ from the earlier package"
    assert np.array_equal(se, reference_se), "SEs differ from the earlier package"
    print("checks passed: normalization; late-label Bayes update; precision inequality; earlier means and SEs")

    # ---- outputs ----
    P_saved = np.zeros((N_TRIALS, n_counts, NP))                                # evidence at the four observation counts
    W_saved = np.zeros((N_TRIALS, n_counts, NP))
    for j in range(n_counts):
        P_saved[:, j] = P_all[:, OBSERVATION_COUNTS[j] - 1]
        W_saved[:, j] = W_all[:, OBSERVATION_COUNTS[j] - 1]
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["observations", "n_trials", "distance_weak", "distance_either", "distance_strong",
                         "strong_minus_weak", "paired_mc_se", "mean_p_present_strong"])
        for j in range(n_counts):
            b = OBSERVATION_COUNTS[j]
            writer.writerow([b, N_TRIALS, distance["W"][:, j].mean(), distance["E"][:, j].mean(), distance["S"][:, j].mean(),
                             mean[j], se[j], p_present_strong[:, j].mean()])
            print(f"{b:2d} observations: Strong - Weak = {mean[j]:.3f} deg (paired SE {se[j]:.3f}); "
                  f"mean P(present | Strong) = {p_present_strong[:, j].mean():.1e}")
    np.savez_compressed(out / "raw.npz", seed=SEED, budgets=OBSERVATION_COUNTS, positions=POS, paths=paths,
                        P=P_saved, W=W_saved, best_weak=best["W"], best_either=best["E"], best_strong=best["S"],
                        distance_weak=distance["W"], distance_either=distance["E"], distance_strong=distance["S"])
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.errorbar(OBSERVATION_COUNTS, mean, yerr=se, marker="o", capsize=4, color="#B55B2D")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set(xticks=OBSERVATION_COUNTS, xlabel="Observations before the late label", ylim=(0, None),
           ylabel="Strong − Weak distance to nearest prior fixation (°)",
           title=f"Same viewing history, different late contrast label\n({N_TRIALS:,} simulated absent trials)")
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.02, "Conditional report: if a target were present, where would it most likely be?\n"
             "Error bars: ±1 paired Monte Carlo SE over trials, not human variability. No new post-label looks.\n"
             "After 8 / 20 looks, mean P(present | Strong) is about 5.6e−6 / 2.4e−18: highly hypothetical reports.\n"
             "Exploratory MAP model with independent looks; AI-assisted illustration, not a human prediction.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.25, top=0.88)
    fig.savefig(out / "figure.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
