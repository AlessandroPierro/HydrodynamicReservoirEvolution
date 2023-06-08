## Optimization of a Hydrodynamic Computational Reservoir through Evolution

To install the needed packages, run the following commands in the terminal:
    
```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```
To run the code for the XNOR task with the amplitude encoding:
    
```bash
    python xnor-amplitude.py --seed 0 --n_workers 12
    python xnor-amplitude.py --seed 1 --n_workers 12
    python xnor-amplitude.py --seed 2 --n_workers 12
    python xnor-amplitude.py --seed 3 --n_workers 12
    python xnor-amplitude.py --seed 4 --n_workers 12
```
To run the code for the XNOR task with the frequency encoding:
    
```bash
    python xnor-frequency.py --seed 0 --n_workers 12
    python xnor-frequency.py --seed 1 --n_workers 12
    python xnor-frequency.py --seed 2 --n_workers 12
    python xnor-frequency.py --seed 3 --n_workers 12
    python xnor-frequency.py --seed 4 --n_workers 12
```
To run the code for the sigmoid regression task with the amplitude encoding:
    
```bash
    python sigmoid-amplitude.py --seed 0 --n_workers 12
    python sigmoid-amplitude.py --seed 1 --n_workers 12
    python sigmoid-amplitude.py --seed 2 --n_workers 12
    python sigmoid-amplitude.py --seed 3 --n_workers 12
    python sigmoid-amplitude.py --seed 4 --n_workers 12
```