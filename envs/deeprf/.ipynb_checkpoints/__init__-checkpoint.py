from gym.envs.registration import register

# Slice-selective excitation pulse
register(
    id='Exc-v51',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.00001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-v0',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.000001,
            'ripple_coef': 0.25,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-v1',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.000001,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)



register(
    id='Exc-v2',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.000005,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-v3',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.00001,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)


register(
    id='Exc-v4',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.00005,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-or-v0',
    entry_point='envs.deeprf.environment:DeepRFSLREXC_origin',
    kwargs={'sar_coef': 0.000001,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-or-v1',
    entry_point='envs.deeprf.environment:DeepRFSLREXC_origin',
    kwargs={'sar_coef': 0.000005,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-or-v2',
    entry_point='envs.deeprf.environment:DeepRFSLREXC_origin',
    kwargs={'sar_coef': 0.00001,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)


register(
    id='Exc-or-v3',
    entry_point='envs.deeprf.environment:DeepRFSLREXC_origin',
    kwargs={'sar_coef': 0.00005,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Exc-or-v4',
    entry_point='envs.deeprf.environment:DeepRFSLREXC_origin',
    kwargs={'sar_coef': 0.0001,
            'ripple_coef': 0.125,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Inv-v0',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_origin',
    kwargs={'sar_coef': 0.0001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Inv-v1',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_OC',
    kwargs={'sar_coef': 0.0003,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Inv-v2',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_origin',
    kwargs={'sar_coef': 0.000001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Inv-v3',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_OC',
    kwargs={'sar_coef': 0.000001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)


register(
    id='Inv-v4',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_origin',
    kwargs={'sar_coef': 0.000005,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Inv-v5',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_OC',
    kwargs={'sar_coef': 0.000005,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)


register(
    id='Inv-v6',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_origin',
    kwargs={'sar_coef': 0.000001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-836, 836, 1400),
            'pos_range2': (-8000, -836, 1400),
            'pos_range3': (836, 8000, 1400)}
)


register(
    id='Inv-v7',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_OC',
    kwargs={'sar_coef': 0.00001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)

register(
    id='Inv-v8',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_origin',
    kwargs={'sar_coef': 0.0000005,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-836, 836, 1400),
            'pos_range2': (-8000, -836, 1400),
            'pos_range3': (836, 8000, 1400)}
)

register(
    id='Inv-v9',
    entry_point='envs.deeprf.environment:DeepRFSLRINV_origin',
    kwargs={'sar_coef': 0.0000005,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-840, 840, 660),
            'pos_range2': (-8000, -840, 670),
            'pos_range3': (840, 8000, 670)}
)
