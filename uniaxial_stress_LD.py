import torch
import numpy as np
from sympy import Symbol, Eq, Function, Number

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.key import Key
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.dataset import DictVariationalDataset
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint, VariationalConstraint
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io import ValidatorPlotter, InferencerPlotter, csv_to_dict
from modulus.sym.loss import Loss
from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node

class largedeformation(PDE):
    name = 'LD'
     
    def __init__(self, E=None, nu=None, lambda_=None, mu=None):
        
        x, y = Symbol('x'), Symbol('y')
        normal_x, normal_y = Symbol('normal_x'), Symbol('normal_y')
        
        input_variables = {'x': x, 'y': y}
        
        u = Function('u')(*input_variables)
        v = Function('v')(*input_variables)
        
        x1 = x + u
        x2 = y + v
        
        # Deformation Gradient
        F11 = u.diff(x) + 1
        F12 = u.diff(y)
        F21 = v.diff(x)
        F22 = v.diff(y) + 1
             
        #  Green-Lagrange Strain
        E11 = 1 / 2 * ((F11 * F11 + F11 * F21) - 1)
        E12 = 1 / 2 * ((F21 * F12 + F21 * F22))
        E21 = 1 / 2 * ((F12 * F11 + F12 * F21))
        E22 = 1 / 2 * ((F22 * F12 + F22 * F22) - 1)
        E33 = - (lambda_ / (2 * mu + lambda_)) * (E11 + E22)
        
        # Compute second Piola-Kirchhoff stress tensor
        S11 = lambda_ * (E11 + E22 + E33) + 2 * mu * E11
        S22 = lambda_ * (E11 + E22 + E33) + 2 * mu * E22
        S12 = mu * E12
        S21 = mu * E21
        
        # Compute First Piola-Kirchhoff stress tensor
        P11 = F11 * S11
        P12 = F12 * S12
        P21 = F21 * S21
        P22 = F22 * S22
               
        # material properties
        if lambda_ is None:
            if isinstance(nu, str):
                nu = Function(nu)(*input_variables)
            elif isinstance(nu, (float, int)):
                nu = Number(nu)
            if isinstance(E, str):
                E = Function(E)(*input_variables)
            elif isinstance(E, (float, int)):
                E = Number(E)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = Function(lambda_)(*input_variables)
            elif isinstance(lambda_, (float, int)):
                lambda_ = Number(lambda_)
            if isinstance(mu, str):
                mu = Function(mu)(*input_variables)
            elif isinstance(mu, (float, int)):
                mu = Number(mu)
                
        # Set Equations
        self.equations = {}
        
        # Equations of equilibrium
        self.equations["equilibrium_x"] = (P11.diff(x) + P12.diff(y))
        self.equations["equilibrium_y"] = (P21.diff(x) + P22.diff(y))

        
        # Traction equations
        self.equations["traction_x"] = normal_x * P11 + normal_y * P12
        self.equations["traction_y"] = normal_x * P21 + normal_y * P22

        

@modulus.sym.main(config_path="./", config_name="config.yaml")
def run(cfg: ModulusConfig) -> None:
    # Parameters
    E = 1000
    nu = 0.3
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu_real = E / (2 * (1 + nu))
    lambda_ = lambda_ / mu_real
    mu = 1.0
    
    domain_origin = (-0.5, -0.5)
    domain_dim = (1, 1)

    # make list of nodes to unroll graph on
    le = largedeformation(lambda_=lambda_, mu=mu, E=E, nu=0.3)
    
    elasticity_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fully_connected,
    )
    
    nodes = le.make_nodes() + [elasticity_net.make_node(name="elasticity_network")]
    
    # domain
    x, y = Symbol('x'), Symbol('y')
    
    square = Rectangle(
        domain_origin,
        (domain_origin[0] + domain_dim[0], domain_origin[1] + domain_dim[1]),
    )
    geo = square

    # make domain
    domain = Domain()
    
# Bottom boundary condition (fixed)
    bottomBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.bottom,
        criteria=Eq(y, domain_origin[1]),
    )
    domain.add_constraint(bottomBC, "bottomBC")

    # Top boundary condition (fixed)
    topBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.top,
        criteria=Eq(y, domain_origin[1] + domain_dim[1]),
    )
    domain.add_constraint(topBC, "topBC")

    # Left boundary condition (traction)
    leftBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": -0.001, "traction_y": 0.0},
        batch_size=cfg.batch_size.left,
        criteria=Eq(x, domain_origin[0]),
    )
    domain.add_constraint(leftBC, "leftBC")

    # Right boundary condition (traction)
    rightBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.001, "traction_y": 0.0},
        batch_size=cfg.batch_size.right,
        criteria=Eq(x, domain_origin[0] + domain_dim[0]),
    )
    domain.add_constraint(rightBC, "rightBC")

    # Interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={'equilibrium_x': 0.0, 'equilibrium_y': 0.0},
        batch_size=cfg.batch_size.interior,
        bounds={x: (domain_origin[0], domain_origin[0] + domain_dim[0]), 
                y: (domain_origin[1], domain_origin[1] + domain_dim[1])},
    )
    domain.add_constraint(interior, 'interior')

    # add inferencer data
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=geo.sample_interior(
            1000,
            bounds={x: (domain_origin[0], domain_origin[0] + domain_dim[0]), 
                    y: (domain_origin[1], domain_origin[1] + domain_dim[1])},
        ),
        output_names=["u", "v"],
        batch_size=1024,
    )
    domain.add_inferencer(inferencer, "inf_data")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()