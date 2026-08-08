from pymatgen.core.structure import Structure
from pymatgen.analysis.adsorption import *
from matplotlib import pyplot as plt
from pymatgen.io.vasp.inputs import Poscar


def show_adsorption_site(data, intermetallic, termination, termination_number, structure_name):
    content = data[intermetallic][termination][termination_number][structure_name]['POSCAR']
    Eads = data[intermetallic][termination][termination_number][structure_name]['Eads']

    filename = "temp_POSCAR"
    with open(filename, "w") as file:
        file.write(content)
    
    adsorption_structure = Structure.from_file(filename)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_slab(adsorption_structure, ax, adsorption_sites=True)
    plt.savefig('figure.png',dpi=300)

    print(f'The predicted H adsorption energy on this site is:{Eads:.2f} eV')


def save_structure(data, intermetallic, termination, termination_number, structure_name):
    content = data[intermetallic][termination][termination_number][structure_name]['POSCAR']

    filename = intermetallic+'_'+termination+'_'+termination_number+'_'+structure_name
    with open(filename, "w") as file:
        file.write(content)
        
    return Structure.from_file(filename)
