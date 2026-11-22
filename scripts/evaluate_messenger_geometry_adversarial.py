#!/usr/bin/env python3
"""Same HEAD-free full-prediction common evaluation; discriminator is not deployed."""
from argparse import ArgumentParser
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.messenger_geometry_adversarial_contract import authenticate
from scripts.evaluate_messenger_geometry_prior import run

if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],required=True)
    run(parser.parse_args(),authenticator=authenticate)
