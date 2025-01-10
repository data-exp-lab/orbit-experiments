"""Console script for running stellar cluster simulations with rebound and random initial conditiosn."""

import unyt
import os
import numpy as np
import rebound
import h5py

import typer
from rich.console import Console
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer()
console = Console()


def get_simulation(bh_mass: float):
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=bh_mass, x=0, y=0, z=0)
    sim.dt = 0.1
    return sim


@app.command()
def run_simulation(
    radius: Annotated[float, typer.Option(help="Radius of the cluster in ly")] = 2.0,
    nstars: Annotated[int, typer.Option(help="Number of stars")] = 25,
    bh_mass: Annotated[float, typer.Option(help="Mass of the black hole")] = 100.0,
    t_final: Annotated[float, typer.Option(help="Final time in years")] = 100.0,
    t_interval: Annotated[
        float, typer.Option(help="Interval of outputs in years")
    ] = 1.0,
    output: Annotated[str, typer.Option(help="Output filename")] = "output.bin",
):
    """Console script for htmdec_formats."""
    console.print(
        f"Creating a star cluster with a central black hole (mass = {bh_mass} Msun) and {nstars} stars, radius of {radius} ly.  Evolving to {t_final} yr."
    )
    radius = radius * unyt.ly
    bh_mass = bh_mass
    t_final = t_final * unyt.yr
    xyz_init = unyt.ly * np.random.normal(
        loc=0.0, scale=radius.in_units("ly").v / 2.5, size=(nstars, 3)
    )
    xyz_init.convert_to_units("AU")
    masses = np.random.normal(loc=2.5, scale=1.0, size=nstars)
    masses = np.maximum(masses, 0.1)
    # I couldn't get set_serialized_data to work, so I'm just adding the stars one by one.
    sim = get_simulation(bh_mass)
    for i in range(nstars):
        sim.add(x=xyz_init[i, 0], y=xyz_init[i, 1], z=xyz_init[i, 2], m=masses[i])
    if os.path.isfile(output):
        os.remove(output)
    sim.save_to_file(output, interval=(t_interval * unyt.yr).in_units("day"))
    console.print("Running...")
    sim.integrate(t_final.in_units("day"))
    console.print(f"Simulation saved to {output}.")


@app.command()
def export_to_hdf5(
    input: Annotated[str, typer.Option(help="File to read from")] = "output.bin",
    output: Annotated[str, typer.Option(help="File to write")] = "output.hdf5",
    shift: Annotated[
        bool, typer.Option(help="Shift to reference frame of Black Hole")
    ] = True,
):
    """Convert a binary output file to an HDF5 file."""
    sims = rebound.Simulationarchive(input)
    with h5py.File(output, "w") as f:
        for i in range(len(sims)):
            sim = sims[i]
            xyz = np.zeros((sim.N, 3))
            masses = np.zeros(sim.N)
            sim.serialize_particle_data(xyz=xyz, m=masses)
            if shift:
                xyz -= xyz[0, :]
            f.create_dataset(f"/sim_{i:05d}/xyz", data=xyz)
            f.create_dataset(f"/sim_{i:05d}/masses", data=masses)
            f[f"/sim_{i:05d}"].attrs["time"] = sim.t


@app.command()
def export_to_csv(
    input: Annotated[str, typer.Option(help="File to read from")] = "output.bin",
    output_prefix: Annotated[
        str, typer.Option(help="Filename prefix for written files")
    ] = "output",
    shift: Annotated[
        bool, typer.Option(help="Shift to reference frame of Black Hole")
    ] = True,
    length_units: Annotated[str, typer.Option(help="Length units for output")] = "AU",
):
    """Convert a binary output file to an HDF5 file."""
    sims = rebound.Simulationarchive(input)
    for i in range(len(sims)):
        sim = sims[i]
        xyz = np.zeros((sim.N, 3)) * unyt.AU
        masses = np.zeros(sim.N)
        sim.serialize_particle_data(xyz=xyz, m=masses)
        xyz.convert_to_units(length_units)
        if shift:
            xyz -= xyz[0, :]
        out = np.concatenate([xyz.d, masses[:, None]], axis=1)
        np.savetxt(
            f"{output_prefix}_{i:05d}.csv",
            out,
            delimiter=",",
            header=f"# x [{length_units}], y [{length_units}], z [{length_units}], mass [Msun]",
            comments=f"## time is {sim.t} days\n",
        )


if __name__ == "__main__":
    app()
