from gym.envs.registration import register

register(
    id='DGW-5x5-Vases-v0',
    entry_point='grid_envs.envs:DGW_Vases_5x5'
)

register(
    id='DynamicGridWorld-v0',
    entry_point='grid_envs.envs:DynamicGridWorld'
)

register(
    id='DynamicGridWorld-2-MovObs-7x7-Random-v0', 
    entry_point='grid_envs.envs:DGW_2_MovObs_7x7_Random'
)

register(
    id='DGW-10x10-Random-RewardTests-v0', 
    entry_point='grid_envs.envs:DGW_10x10_Random_RewardTests'
)

register(
    id='DGW-10x10-Random-Simple-v0', 
    entry_point='grid_envs.envs:DGW_10x10_Random_Simple'
)

register(
    id='DGW-10x10-Random-Vases-v0', 
    entry_point='grid_envs.envs:DGW_10x10_Random_Vases'
)

register(
    id='DGW-10x10-Random-Pillars-v0', 
    entry_point='grid_envs.envs:DGW_10x10_Random_Pillars'
)

register(
    id='DGW-10x10-Random-Hazards-v0', 
    entry_point='grid_envs.envs:DGW_10x10_Random_Hazards'
)

register(
    id='DGW-10x10-Random-Multi-v0', 
    entry_point='grid_envs.envs:DGW_10x10_Random_Multi'
)


register(
    id='DGW-LeftRight-v0', 
    entry_point='grid_envs.envs:DGW_LeftRight'
)

register(
    id='DGW-UpDown-v0', 
    entry_point='grid_envs.envs:DGW_UpDown'
)

register(
    id='DGW-MovObs-WallRandom-Multi-v0', 
    entry_point='grid_envs.envs:DGW_MovObs_WallRandom_Multi'
)

register(
    id='DGW-WallRandom-Multi-v0', 
    entry_point='grid_envs.envs:DGW_WallRandom_Multi'
)

register(
    id='DGW-WallRandom-v0', 
    entry_point='grid_envs.envs:DGW_WallRandom'
)

register(
    id='DGW-WallTop-v0', 
    entry_point='grid_envs.envs:DGW_WallTop'
)

register(
    id='DGW-WallBottom-v0', 
    entry_point='grid_envs.envs:DGW_WallBottom'
)

register(
    id='DGW-MovingPillars-v0', 
    entry_point='grid_envs.envs:DGW_MovingPillars'
)

register(
    id='DGW-MovingVases-v0', 
    entry_point='grid_envs.envs:DGW_MovingVases'
)

##################################
### stable benchmark scenarios ###
##################################

### LEVEL 0 ###
register(
    id='DGW-lvl-0-v0', 
    entry_point='grid_envs.envs:DGW_lvl_0'
)

### LEVEL 1 ###
register(
    id='DGW-lvl-1-hazards-v0', 
    entry_point='grid_envs.envs:DGW_lvl_1_hazards'
)

register(
    id='DGW-lvl-1-vases-v0', 
    entry_point='grid_envs.envs:DGW_lvl_1_vases'
)

register(
    id='DGW-lvl-1-pillars-v0', 
    entry_point='grid_envs.envs:DGW_lvl_1_pillars'
)
register(
    id='DGW-lvl-1-mixed-v0', 
    entry_point='grid_envs.envs:DGW_lvl_1_mixed'
)

### LEVEL 2 ###
register(
    id='DGW-lvl-2-hazards-v0', 
    entry_point='grid_envs.envs:DGW_lvl_2_hazards'
)

register(
    id='DGW-lvl-2-vases-v0', 
    entry_point='grid_envs.envs:DGW_lvl_2_vases'
)

register(
    id='DGW-lvl-2-pillars-v0', 
    entry_point='grid_envs.envs:DGW_lvl_2_pillars'
)
register(
    id='DGW-lvl-2-mixed-v0', 
    entry_point='grid_envs.envs:DGW_lvl_2_mixed'
)

