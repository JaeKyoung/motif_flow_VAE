from setuptools import setup

setup(
    name="se3_flow_vae",
    packages=[
        'motifflow',
        #'data_temp',
        'ProteinMPNN',
        'openfold',
    ],
    package_dir={
        'motifflow': './motifflow',
	    #'data_temp': './data_temp',
        'ProteinMPNN': './ProteinMPNN',
        'openfold': './openfold',
    },
)
