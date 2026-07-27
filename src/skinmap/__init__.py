"""skinmap — 3D skin-scan registration, artifact checking, and mole tracking.

The vision half of the openderm project: everything that turns a
captured scan folder (``captures/<scan>/``) into a registered 3D
reconstruction and tracks lesions across scans. Runs on the workstation
(needs opencv/scipy — install with ``pip install -e ".[vision]"``), unlike
the ``openderm`` package which drives the hardware on the Pis.

The supported workflow uses ``openderm-process`` for reconstruction and
artifact checks and ``openderm-compare`` for longitudinal comparison. See the
root README and ``docs/skin-registration.md`` for usage.
"""
