The conda package specification includes the package name (i.e. singa), version and build string (could be very long).
To install a certain Singa package, we run

    conda install -c nusdbsystem singa=<version>=<build string>

It is inconvenient to type all 3 parts when running the installation commands.
The meta.yml file in this folder is to create a conda package `singa-cpu` as
an alias of one specific Singa package.
It does nothing except creating a dummy conda package that depends on one real
cpu version singa package.  For example, the following line in meta.yml indicates
that singa-cpu depends on singa with version 1.1.1, python version=3.6

    - singa 1.1.1 py36_cpu

Therefore, when we run

    conda install -c nusdbsystem singa-cpu

The dependent Singa package will be installed.
By default, singa-cpu depends on the latest Singa (py3.6).
When we have a new Singa version available, we need to update the meta.yml file to
change the dependency.

To build this package and upload it

    conda config --add channels nusdbsystem
    conda build .
    anaconda -t $ANACONDA_UPLOAD_TOKEN upload -u nusdbsystem -l main <path to the singa-cpu package>

where $ANACONDA_UPLOAD_TOKEN is the upload token associated with nusdbsystem account on anaconda cloud.
