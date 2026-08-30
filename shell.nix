# Dev environment for CombinatorialProbability.
#
# pip/venv is not usable on this NixOS box: the manylinux numpy and scipy
# wheels cannot find libstdc++.so.6, so they import-error at runtime. Nix
# supplies the same libraries already linked.
#
#   nix-shell          # then: python, pytest, ipython
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      numpy scipy matplotlib sympy ipython pytest jupyter
    ]))
  ];
  shellHook = ''
    export PYTHONPATH="$(dirname "$PWD")''${PYTHONPATH:+:$PYTHONPATH}"
    echo "CombinatorialProbability dev shell — import as: from CombinatorialProbability import IntegerPartition"
  '';
}
