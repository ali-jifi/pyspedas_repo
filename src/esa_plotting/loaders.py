from typing import Iterable

import pyspedas
from pyspedas.projects import themis


def load_esa(
    probes: str | Iterable[str],
    trange: tuple[str, str] | list[str],
    level: str = "l2",
    time_clip: bool = True,
) -> list[str]:
    return themis.esa(
        probe=list(probes) if not isinstance(probes, str) else probes,
        trange=list(trange),
        level=level,
        notplot=False,
        time_clip=time_clip,
    )


def load_esd(
    probe: str,
    trange: tuple[str, str] | list[str],
    datatype: str = "peif",
    time_clip: bool = True,
) -> list[str]:
    return themis.esd(
        probe=probe,
        trange=list(trange),
        datatype=datatype,
        time_clip=time_clip,
    )


def load_fgm(
    probe: str,
    trange: tuple[str, str] | list[str],
    level: str = "l2",
    time_clip: bool = True,
) -> list[str]:
    return themis.fgm(
        probe=probe,
        trange=list(trange),
        level=level,
        time_clip=time_clip,
    )


def pyspedas_version() -> str:
    return pyspedas.version()
