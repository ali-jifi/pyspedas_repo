# ion beam detection pipeline for themis esa data

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cdflib import CDF
from pyspedas import get_data
from pyspedas.projects import themis
from scipy.interpolate import interp1d


@dataclass
class ESDDistribution:
    # 3d ion dist from themis esa esd files
    times: np.ndarray          # (ntime,) unix timestamps
    eflux: np.ndarray          # (ntime, 32, 176) energy flux
    energy: np.ndarray         # (32,) energy vals in ev for active mode
    theta: np.ndarray          # (32, 176) polar angle in instrument coords, deg
    phi: np.ndarray            # (32, 176) azimuthal angle in instrument coords, deg
    domega: np.ndarray         # (32, 176) solid angle element
    bins_mask: np.ndarray      # (ntime, 32, 176) valid bin mask
    phi_offset: np.ndarray     # (ntime,) spin-phase correction
    en_ind: np.ndarray         # (ntime,) energy mode idx per timestep
    an_ind: np.ndarray         # (ntime,) angle mode idx per timestep


def load_esd_distribution(probe: str, trange: list[str], data_dir: str) -> ESDDistribution:
    # reads cdf directly cuz pyspedas doesnt expose angle/energy lookup tables
    import os
    os.environ["THM_DATA_DIR"] = data_dir

    themis.esd(probe=probe, trange=trange, datatype="peif",
               time_clip=True, downloadonly=True)

    pattern = str(
        Path(data_dir)
        / f"th{probe}" / "l2" / "esd" / "*"
        / f"th{probe}_l2_esa_peif_*.cdf"
    )
    cdf_files = sorted(glob.glob(pattern))
    if not cdf_files:
        raise FileNotFoundError(f"No ESD CDF files found matching {pattern}")

    all_times, all_eflux, all_bins = [], [], []
    all_en_ind, all_an_ind, all_phi_offset = [], [], []
    energy_table = phi_table = theta_table = domega_table = None

    for cdf_path in cdf_files:
        cdf = CDF(cdf_path)

        epoch = cdf.varget("epoch")
        from cdflib.epochs import CDFepoch
        unix_times = CDFepoch.unixtime(epoch)

        t0_str, t1_str = trange
        from datetime import datetime, timezone
        t0_unix = datetime.strptime(t0_str.split("/")[0], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
        if "/" in t0_str:
            parts = t0_str.split("/")[1].split(":")
            t0_unix += int(parts[0]) * 3600
            if len(parts) > 1:
                t0_unix += int(parts[1]) * 60
            if len(parts) > 2:
                t0_unix += float(parts[2])

        t1_unix = datetime.strptime(t1_str.split("/")[0], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
        if "/" in t1_str:
            parts = t1_str.split("/")[1].split(":")
            t1_unix += int(parts[0]) * 3600
            if len(parts) > 1:
                t1_unix += int(parts[1]) * 60
            if len(parts) > 2:
                t1_unix += float(parts[2])

        mask = (unix_times >= t0_unix) & (unix_times <= t1_unix)
        if not np.any(mask):
            continue

        unix_times = np.asarray(unix_times)[mask]
        eflux = cdf.varget("eflux")[mask]
        bins_data = cdf.varget("bins")[mask]
        en_ind = cdf.varget("en_ind")[mask]
        an_ind = cdf.varget("an_ind")[mask]
        phi_offset_data = cdf.varget("phi_offset")[mask]

        all_times.append(unix_times)
        all_eflux.append(eflux)
        all_bins.append(bins_data)
        all_en_ind.append(en_ind)
        all_an_ind.append(an_ind)
        all_phi_offset.append(phi_offset_data)

        if energy_table is None:
            energy_table = cdf.varget("energy")     # (32, 5)
            theta_table = cdf.varget("theta")        # (32, 176, 3)
            phi_table = cdf.varget("phi")             # (32, 176, 3)
            domega_table = cdf.varget("domega")       # (32, 176, 3)

        del cdf

    if not all_times:
        raise ValueError(f"No ESD data in time range {trange}")

    times = np.concatenate(all_times)
    eflux = np.concatenate(all_eflux)
    bins_mask = np.concatenate(all_bins)
    en_ind = np.concatenate(all_en_ind)
    an_ind = np.concatenate(all_an_ind)
    phi_offset = np.concatenate(all_phi_offset)

    mode_en = int(np.median(en_ind))
    mode_an = int(np.median(an_ind))
    energy = energy_table[:, mode_en]
    theta = theta_table[:, :, mode_an]
    phi = phi_table[:, :, mode_an]
    domega = domega_table[:, :, mode_an]

    return ESDDistribution(
        times=times, eflux=eflux, energy=energy,
        theta=theta, phi=phi, domega=domega,
        bins_mask=bins_mask, phi_offset=phi_offset,
        en_ind=en_ind, an_ind=an_ind,
    )


def load_bfield_dsl(probe: str, trange: list[str], data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    # returns (times, b_dsl) where b_dsl is (n,3)
    import os
    os.environ["THM_DATA_DIR"] = data_dir
    themis.fgm(probe=probe, trange=trange, level="l2", time_clip=True)
    varname = f"th{probe}_fgs_dsl"
    d = get_data(varname)
    if d is None:
        varname = f"th{probe}_fgl_dsl"
        d = get_data(varname)
    if d is None:
        varname = f"th{probe}_fgh_dsl"
        d = get_data(varname)
    if d is None:
        raise ValueError(f"No FGM DSL data found for probe {probe}")
    return d.times, d.y


def load_moments(probe: str, trange: list[str], data_dir: str) -> dict:
    import os
    os.environ["THM_DATA_DIR"] = data_dir
    themis.esa(probe=probe, trange=trange, level="l2", time_clip=True)

    prefix = f"th{probe}_peif"
    result = {}

    d = get_data(f"{prefix}_density")
    if d is not None:
        result["density_times"] = d.times
        result["density"] = d.y

    d = get_data(f"{prefix}_velocity_gsm")
    if d is None:
        d = get_data(f"{prefix}_velocity_dsl")
    if d is not None:
        result["velocity_times"] = d.times
        result["velocity"] = d.y

    d = get_data(f"{prefix}_avgtemp")
    if d is not None:
        result["temp_times"] = d.times
        result["temperature"] = d.y

    d = get_data(f"{prefix}_vthermal")
    if d is not None:
        result["vthermal"] = d.y

    return result


@dataclass
class PitchAngleSpectra:
    # pa-gated energy spectra over a time interval
    times: np.ndarray          # (ntime,)
    energy: np.ndarray         # (nenergy,) energy bin centers in ev
    omni: np.ndarray           # (ntime, nenergy) omnidirectional flux
    para: np.ndarray           # (ntime, nenergy) field-aligned 0-30 deg
    anti: np.ndarray           # (ntime, nenergy) anti-field-aligned 150-180 deg
    pa_coverage_para: np.ndarray  # (ntime,) frac of solid angle in para cone
    pa_coverage_anti: np.ndarray  # (ntime,) frac of solid angle in anti cone


def _compute_pitch_angles(theta_inst: np.ndarray, phi_inst: np.ndarray,
                          b_dsl: np.ndarray, phi_offset: float) -> np.ndarray:
    # pa btwn particle vel and b, vel is opposite to look dir
    theta_rad = np.deg2rad(theta_inst)
    phi_rad = np.deg2rad(phi_inst + phi_offset)

    look_x = np.cos(theta_rad) * np.cos(phi_rad)
    look_y = np.cos(theta_rad) * np.sin(phi_rad)
    look_z = np.sin(theta_rad)

    # particle vel is opposite look dir
    vx, vy, vz = -look_x, -look_y, -look_z

    b_mag = np.linalg.norm(b_dsl)
    if b_mag < 0.1:
        return np.full_like(theta_inst, np.nan)

    bhat = b_dsl / b_mag
    cos_pa = vx * bhat[0] + vy * bhat[1] + vz * bhat[2]
    cos_pa = np.clip(cos_pa, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_pa))


def compute_pa_spectra(
    dist: ESDDistribution,
    b_times: np.ndarray,
    b_dsl: np.ndarray,
    para_range: tuple[float, float] = (0.0, 30.0),
    anti_range: tuple[float, float] = (150.0, 180.0),
) -> PitchAngleSpectra:
    # reduces 3d dist to pa-gated energy spectra per timestep
    b_interp = interp1d(b_times, b_dsl, axis=0, kind="linear",
                        bounds_error=False, fill_value="extrapolate")

    ntime = len(dist.times)
    nenergy = len(dist.energy)
    valid_energy_mask = dist.energy > 0

    omni = np.full((ntime, nenergy), np.nan)
    para = np.full((ntime, nenergy), np.nan)
    anti = np.full((ntime, nenergy), np.nan)
    cov_para = np.zeros(ntime)
    cov_anti = np.zeros(ntime)

    for t in range(ntime):
        b_vec = b_interp(dist.times[t])
        phi_off = dist.phi_offset[t]
        bins_valid = dist.bins_mask[t]
        flux = dist.eflux[t]

        for e in range(nenergy):
            if not valid_energy_mask[e]:
                continue

            valid = bins_valid[e].astype(bool)
            if not np.any(valid):
                continue

            f = flux[e, valid]
            dw = dist.domega[e, valid]

            th_bins = dist.theta[e, valid]
            ph_bins = dist.phi[e, valid]

            pa = _compute_pitch_angles(th_bins, ph_bins, b_vec, phi_off)

            nan_mask = np.isnan(pa)
            if np.all(nan_mask):
                continue

            good = ~nan_mask & np.isfinite(f) & (f > 0)
            if not np.any(good):
                continue

            total_weight = dw[good].sum()
            if total_weight > 0:
                omni[t, e] = np.sum(f[good] * dw[good]) / total_weight

            in_para = good & (pa >= para_range[0]) & (pa <= para_range[1])
            if np.any(in_para):
                w_para = dw[in_para].sum()
                para[t, e] = np.sum(f[in_para] * dw[in_para]) / w_para
                cov_para[t] = max(cov_para[t], w_para / total_weight)

            in_anti = good & (pa >= anti_range[0]) & (pa <= anti_range[1])
            if np.any(in_anti):
                w_anti = dw[in_anti].sum()
                anti[t, e] = np.sum(f[in_anti] * dw[in_anti]) / w_anti
                cov_anti[t] = max(cov_anti[t], w_anti / total_weight)

    return PitchAngleSpectra(
        times=dist.times,
        energy=dist.energy,
        omni=omni,
        para=para,
        anti=anti,
        pa_coverage_para=cov_para,
        pa_coverage_anti=cov_anti,
    )


@dataclass
class FeatureTable:
    # per-timestep features for beam classification
    times: np.ndarray
    e_peak: np.ndarray           # energy of max flux, ev
    width: np.ndarray            # normalized spectral width
    asymmetry: np.ndarray        # (f_para - f_anti) / (f_para + f_anti) near e_peak
    para_to_omni: np.ndarray     # parallel cone / omni ratio near e_peak
    energy_ratio: np.ndarray     # e_flow / e_th from moments
    pa_coverage_ok: np.ndarray   # bool, both cones adequately sampled


PROTON_MASS_KG = 1.6726219e-27
EV_PER_JOULE = 6.242e18
BOLTZMANN_EV = 8.617e-5


def extract_features(
    spectra: PitchAngleSpectra,
    moments: dict,
    energy_cutoff_low: float = 30.0,
    pa_coverage_threshold: float = 0.01,
) -> FeatureTable:
    ntime = len(spectra.times)
    energy = spectra.energy

    valid_e = energy >= energy_cutoff_low
    e_valid = energy[valid_e]

    e_peak = np.full(ntime, np.nan)
    width = np.full(ntime, np.nan)
    asymmetry = np.full(ntime, np.nan)
    para_to_omni = np.full(ntime, np.nan)
    energy_ratio = np.full(ntime, np.nan)
    pa_ok = np.zeros(ntime, dtype=bool)

    if "velocity" in moments and "temperature" in moments:
        vel_interp = interp1d(moments["velocity_times"], moments["velocity"],
                              axis=0, kind="nearest",
                              bounds_error=False, fill_value=np.nan)
        temp_interp = interp1d(moments["temp_times"], moments["temperature"],
                               axis=0, kind="nearest",
                               bounds_error=False, fill_value=np.nan)
    else:
        vel_interp = temp_interp = None

    for t in range(ntime):
        omni_t = spectra.omni[t, valid_e]
        para_t = spectra.para[t, valid_e]
        anti_t = spectra.anti[t, valid_e]

        pa_ok[t] = (spectra.pa_coverage_para[t] >= pa_coverage_threshold and
                    spectra.pa_coverage_anti[t] >= pa_coverage_threshold)

        if np.all(np.isnan(omni_t)):
            continue
        omni_finite = np.where(np.isfinite(omni_t), omni_t, 0.0)
        idx_peak = np.argmax(omni_finite)
        e_peak[t] = e_valid[idx_peak]

        # flux-weighted energy spread normalized by e_peak
        total_flux = np.nansum(omni_finite)
        if total_flux > 0 and e_peak[t] > 0:
            e_mean = np.nansum(omni_finite * e_valid) / total_flux
            e_var = np.nansum(omni_finite * (e_valid - e_mean) ** 2) / total_flux
            width[t] = np.sqrt(e_var) / e_peak[t]

        # +/-2 energy bins around peak for asymmetry window
        lo = max(0, idx_peak - 2)
        hi = min(len(e_valid), idx_peak + 3)
        f_para = np.nanmean(para_t[lo:hi])
        f_anti = np.nanmean(anti_t[lo:hi])

        if np.isfinite(f_para) and np.isfinite(f_anti) and (f_para + f_anti) > 0:
            asymmetry[t] = (f_para - f_anti) / (f_para + f_anti)

        f_omni = np.nanmean(omni_finite[lo:hi])
        if np.isfinite(f_para) and f_omni > 0:
            para_to_omni[t] = f_para / f_omni

        # e_flow = 0.5 * m * v^2 in ev
        if vel_interp is not None and temp_interp is not None:
            v = vel_interp(spectra.times[t])
            T = temp_interp(spectra.times[t])
            if np.all(np.isfinite(v)) and np.isfinite(T) and T > 0:
                v_mag = np.linalg.norm(v)
                e_flow = 0.5 * PROTON_MASS_KG * (v_mag * 1e3) ** 2 * EV_PER_JOULE
                energy_ratio[t] = e_flow / T

    return FeatureTable(
        times=spectra.times,
        e_peak=e_peak,
        width=width,
        asymmetry=asymmetry,
        para_to_omni=para_to_omni,
        energy_ratio=energy_ratio,
        pa_coverage_ok=pa_ok,
    )


@dataclass
class ClassifierParams:
    # spectral features are primary, moments are soft cuz they mix beam + plasma sheet
    asymmetry_min: float = 0.2
    width_max: float = 0.8
    para_to_omni_min: float = 1.3
    energy_ratio_min: float = 0.5
    score_threshold: float = 0.4
    # weights, spectral > moments
    w_asymmetry: float = 0.35
    w_width: float = 0.25
    w_para_to_omni: float = 0.25
    w_energy_ratio: float = 0.15


@dataclass
class ClassificationResult:
    times: np.ndarray
    is_beam: np.ndarray          # bool
    beam_score: np.ndarray       # 0-1 continuous score
    beam_direction: np.ndarray   # +1 parallel, -1 anti-parallel, 0 unknown


def classify_beams(features: FeatureTable,
                   params: ClassifierParams | None = None) -> ClassificationResult:
    if params is None:
        params = ClassifierParams()

    ntime = len(features.times)
    is_beam = np.zeros(ntime, dtype=bool)
    beam_score = np.zeros(ntime)
    beam_direction = np.zeros(ntime, dtype=int)

    for t in range(ntime):
        er = features.energy_ratio[t]
        asym = features.asymmetry[t]
        w = features.width[t]
        p2o = features.para_to_omni[t]

        if not features.pa_coverage_ok[t]:
            continue

        # component scores, 0 to 1
        s_asym = 0.0
        if np.isfinite(asym):
            s_asym = np.clip(abs(asym) / params.asymmetry_min, 0, 2) / 2

        s_width = 0.0
        if np.isfinite(w):
            s_width = np.clip((params.width_max - w) / params.width_max, 0, 1)

        s_p2o = 0.0
        if np.isfinite(p2o):
            s_p2o = np.clip((p2o - 1.0) / (params.para_to_omni_min - 1.0), 0, 2) / 2

        s_er = 0.0
        if np.isfinite(er):
            s_er = np.clip(er / params.energy_ratio_min, 0, 2) / 2

        beam_score[t] = (params.w_asymmetry * s_asym +
                         params.w_width * s_width +
                         params.w_para_to_omni * s_p2o +
                         params.w_energy_ratio * s_er)

        # score-based, catches weak beams spread across features
        score_ok = beam_score[t] >= params.score_threshold

        # hard rule fallback, catches strong beams even if score is borderline
        asym_ok = np.isfinite(asym) and abs(asym) >= params.asymmetry_min
        width_ok = np.isfinite(w) and w <= params.width_max
        p2o_ok = np.isfinite(p2o) and p2o >= params.para_to_omni_min
        hard_ok = asym_ok and (width_ok or p2o_ok)

        is_beam[t] = score_ok or hard_ok

        if np.isfinite(asym):
            if asym > 0:
                beam_direction[t] = 1
            elif asym < 0:
                beam_direction[t] = -1

    return ClassificationResult(
        times=features.times,
        is_beam=is_beam,
        beam_score=beam_score,
        beam_direction=beam_direction,
    )


def smooth_labels(result: ClassificationResult,
                  min_consecutive: int = 2) -> ClassificationResult:
    # requires min_consecutive flagged steps to keep a beam interval
    smoothed = np.zeros_like(result.is_beam)
    n = len(smoothed)

    run_start = None
    for i in range(n + 1):
        if i < n and result.is_beam[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_consecutive:
                    smoothed[run_start:i] = True
                run_start = None

    return ClassificationResult(
        times=result.times,
        is_beam=smoothed,
        beam_score=result.beam_score,
        beam_direction=result.beam_direction,
    )


def threshold_sensitivity(features: FeatureTable,
                          param_ranges: dict | None = None) -> dict:
    # varies thresholds, reports label stability
    if param_ranges is None:
        param_ranges = {
            "asymmetry_min": [0.1, 0.15, 0.2, 0.3, 0.4],
            "width_max": [0.5, 0.6, 0.8, 1.0, 1.2],
            "para_to_omni_min": [1.1, 1.2, 1.3, 1.5, 1.8],
        }

    results = {}
    base = ClassifierParams()
    base_result = classify_beams(features, base)
    base_count = base_result.is_beam.sum()

    for param_name, values in param_ranges.items():
        counts = []
        for val in values:
            kwargs = {
                "asymmetry_min": base.asymmetry_min,
                "width_max": base.width_max,
                "para_to_omni_min": base.para_to_omni_min,
                "energy_ratio_min": base.energy_ratio_min,
            }
            kwargs[param_name] = val
            p = ClassifierParams(**kwargs)
            r = classify_beams(features, p)
            counts.append(int(r.is_beam.sum()))
        results[param_name] = {"values": values, "beam_counts": counts,
                               "base_count": int(base_count)}

    return results


def plot_spectra_snapshot(
    spectra: PitchAngleSpectra,
    time_idx: int,
    label: str = "",
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    e = spectra.energy
    valid = e > 0

    ax.loglog(e[valid], spectra.omni[time_idx, valid], "k-", lw=2, label="Omni")
    ax.loglog(e[valid], spectra.para[time_idx, valid], "r-", lw=1.5, label="0°–30° (para)")
    ax.loglog(e[valid], spectra.anti[time_idx, valid], "b-", lw=1.5, label="150°–180° (anti)")

    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Energy Flux [eV/cm²-s-sr-eV]")
    ax.legend(fontsize=9)
    if label:
        ax.set_title(label)
    ax.grid(True, alpha=0.3)
    return ax


def plot_feature_timeseries(
    features: FeatureTable,
    classification: ClassificationResult,
    spectra: PitchAngleSpectra,
    out_png: str,
    title: str = "",
):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timezone

    times_dt = [datetime.fromtimestamp(t, tz=timezone.utc) for t in features.times]

    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1, 0.5]})

    # omni spectrogram
    ax = axes[0]
    e_valid = spectra.energy > 0
    e_plot = spectra.energy[e_valid]
    z = spectra.omni[:, e_valid].T
    z = np.where(z > 0, z, np.nan)
    pcm = ax.pcolormesh(times_dt, e_plot, z,
                        norm=plt.matplotlib.colors.LogNorm(vmin=1e3, vmax=1e8),
                        cmap="jet", shading="auto")
    ax.set_yscale("log")
    ax.set_ylabel("Energy [eV]")
    ax.set_ylim(5, 30000)
    fig.colorbar(pcm, ax=ax, label="eflux", pad=0.01)
    if title:
        ax.set_title(title)

    axes[1].semilogy(times_dt, features.e_peak, "k.", ms=2)
    axes[1].set_ylabel("E_peak [eV]")
    axes[1].set_ylim(10, 30000)

    axes[2].plot(times_dt, features.width, "k.", ms=2)
    axes[2].set_ylabel("Width")
    axes[2].axhline(0.8, color="r", ls="--", alpha=0.5, label="threshold")
    axes[2].legend(fontsize=8)

    axes[3].plot(times_dt, features.asymmetry, "k.", ms=2)
    axes[3].set_ylabel("Asymmetry")
    axes[3].axhline(0.2, color="r", ls="--", alpha=0.5)
    axes[3].axhline(-0.2, color="r", ls="--", alpha=0.5)
    axes[3].set_ylim(-1.1, 1.1)

    axes[4].semilogy(times_dt, features.energy_ratio, "k.", ms=2)
    axes[4].set_ylabel("E_flow/E_th")
    axes[4].axhline(0.5, color="r", ls="--", alpha=0.5)

    axes[5].plot(times_dt, classification.beam_score, "k.", ms=2)
    axes[5].set_ylabel("Beam Score")
    axes[5].set_ylim(0, 1.1)

    # classification color bar
    ax = axes[6]
    colors = []
    for i in range(len(features.times)):
        if classification.is_beam[i]:
            if classification.beam_direction[i] > 0:
                colors.append("red")
            elif classification.beam_direction[i] < 0:
                colors.append("blue")
            else:
                colors.append("orange")
        else:
            colors.append("lightgray")

    for i, (t, c) in enumerate(zip(times_dt, colors)):
        ax.axvspan(t, times_dt[min(i + 1, len(times_dt) - 1)],
                   color=c, alpha=0.8)
    ax.set_ylabel("Beam")
    ax.set_yticks([])

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("UT")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


def plot_curated_snapshots(
    spectra: PitchAngleSpectra,
    classification: ClassificationResult,
    features: FeatureTable,
    out_png: str,
    n_beam: int = 3,
    n_ps: int = 3,
    include_borderline: bool = True,
):
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone

    beam_idx = np.where(classification.is_beam)[0]
    ps_idx = np.where(~classification.is_beam & features.pa_coverage_ok)[0]

    selected = []
    labels = []

    if len(beam_idx) > 0:
        step = max(1, len(beam_idx) // n_beam)
        for i in beam_idx[::step][:n_beam]:
            selected.append(i)
            t = datetime.fromtimestamp(spectra.times[i], tz=timezone.utc)
            labels.append(f"BEAM {t.strftime('%H:%M:%S')}")

    if len(ps_idx) > 0:
        step = max(1, len(ps_idx) // n_ps)
        for i in ps_idx[::step][:n_ps]:
            selected.append(i)
            t = datetime.fromtimestamp(spectra.times[i], tz=timezone.utc)
            labels.append(f"PS {t.strftime('%H:%M:%S')}")

    if include_borderline:
        borderline = np.where(
            (classification.beam_score > 0.3) &
            (classification.beam_score < 0.7) &
            features.pa_coverage_ok
        )[0]
        if len(borderline) > 0:
            step = max(1, len(borderline) // 2)
            for i in borderline[::step][:2]:
                if i not in selected:
                    selected.append(i)
                    t = datetime.fromtimestamp(spectra.times[i], tz=timezone.utc)
                    labels.append(f"BORDER {t.strftime('%H:%M:%S')}")

    if not selected:
        print("[WARN] No timesteps selected for snapshot plot")
        return

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if len(selected) == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (sel, lbl) in enumerate(zip(selected, labels)):
        r, c = divmod(idx, ncols)
        plot_spectra_snapshot(spectra, sel, label=lbl, ax=axes[r, c])

    for idx in range(len(selected), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


@dataclass
class PipelineResult:
    spectra: PitchAngleSpectra
    features: FeatureTable
    classification: ClassificationResult
    classification_smoothed: ClassificationResult
    sensitivity: dict = field(default_factory=dict)


def run_pipeline(
    probe: str,
    trange: list[str],
    data_dir: str,
    params: ClassifierParams | None = None,
    min_consecutive: int = 2,
    energy_cutoff_low: float = 30.0,
    figures_dir: str | None = None,
) -> PipelineResult:
    print(f"=== Phase 0: Loading data for THEMIS-{probe.upper()} {trange} ===")
    dist = load_esd_distribution(probe, trange, data_dir)
    print(f"  ESD: {len(dist.times)} timesteps, {len(dist.energy)} energy bins")

    b_times, b_dsl = load_bfield_dsl(probe, trange, data_dir)
    print(f"  FGM: {len(b_times)} samples")

    moments = load_moments(probe, trange, data_dir)
    print(f"  Moments: density={len(moments.get('density', []))}, "
          f"velocity={len(moments.get('velocity', []))}")

    print(f"\n=== Phase 1: Pitch-angle spectral reduction ===")
    spectra = compute_pa_spectra(dist, b_times, b_dsl)
    n_valid_para = np.sum(spectra.pa_coverage_para > 0)
    n_valid_anti = np.sum(spectra.pa_coverage_anti > 0)
    print(f"  Para coverage: {n_valid_para}/{len(spectra.times)} timesteps")
    print(f"  Anti coverage: {n_valid_anti}/{len(spectra.times)} timesteps")

    print(f"\n=== Phase 2: Feature extraction ===")
    features = extract_features(spectra, moments,
                                energy_cutoff_low=energy_cutoff_low)
    n_finite = np.sum(np.isfinite(features.asymmetry))
    print(f"  Features computed: {n_finite}/{len(features.times)} with valid asymmetry")

    print(f"\n=== Phase 3: Classification ===")
    classification = classify_beams(features, params)
    n_beam = classification.is_beam.sum()
    print(f"  Raw beams: {n_beam}/{len(features.times)} timesteps")

    print(f"\n=== Phase 5: Temporal smoothing ===")
    smoothed = smooth_labels(classification, min_consecutive=min_consecutive)
    n_smooth = smoothed.is_beam.sum()
    print(f"  Smoothed beams: {n_smooth}/{len(features.times)} timesteps")

    sensitivity = threshold_sensitivity(features)
    print(f"  Sensitivity analysis complete")

    if figures_dir is not None:
        print(f"\n=== Phase 4: Plotting ===")
        fig_dir = Path(figures_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        date_str = trange[0].split("/")[0].replace("-", "")
        prefix = f"th{probe}_beam_{date_str}"

        plot_feature_timeseries(
            features, smoothed, spectra,
            str(fig_dir / f"{prefix}_overview.png"),
            title=f"THEMIS-{probe.upper()} Beam Detection {trange[0]}",
        )

        plot_curated_snapshots(
            spectra, smoothed, features,
            str(fig_dir / f"{prefix}_snapshots.png"),
        )

    return PipelineResult(
        spectra=spectra,
        features=features,
        classification=classification,
        classification_smoothed=smoothed,
        sensitivity=sensitivity,
    )
