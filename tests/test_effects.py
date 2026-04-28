from sepfp.data.effects import RandomizedEffectChain


def test_randomized_effect_chain_accepts_hydra_style_target_mapping():
    chain = RandomizedEffectChain(
        [
            {
                "effect": {"_target_": "pedalboard.PeakFilter"},
                "p": 0.2,
                "cutoff_frequency_hz": "uniform 440 50 8000",
                "gain_db": "uniform 0 -20 10",
                "q": "random 1. 0. 1.",
            }
        ]
    )
    params = chain.sample_parameters()
    assert params.ops[0].name == "PeakFilter"
