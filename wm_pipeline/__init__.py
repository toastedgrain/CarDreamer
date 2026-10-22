"""
wm_pipeline: our Stage 1-3 pipeline on top of CarDreamer + DreamerV3.

    (O_t, l)     --Stage 1: observation-->  Observation
    Observation  --Stage 2: encoder E-->    latent state x_t
    x_t          --Stage 3: actor pi-->     nominal action u_nom
    u_nom        --safety filter-->         u_safe  -->  env.step(u_safe)
                   (identity for now, CBF later)

Rule for this package: importing it must never import `carla` or `car_dreamer`.
Both need a CARLA install, and Phase A runs on a laptop without one. Code that talks
to the real environment imports them inside the function that needs them.
"""
