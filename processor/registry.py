"""CLI processor registrations, grouped by maintenance status."""

# CLI entry points; Processor and PT_Processor are shared base classes.
# Store import paths to avoid circular imports and loading training dependencies.
ACTIVE_PROCESSORS = {
    'linear_evaluation': 'processor.linear_evaluation.LE_Processor',
    'pretrain_skeletonclr': 'processor.pretrain_skeletonclr.SkeletonCLR_Processor',
    'pretrain_skeletonclr_3views': 'processor.pretrain_skeletonclr_3views.SkeletonCLR_3views_Processor',
    'plot_skeletonclr': 'processor.plot_skeletonclr.SkeletonCLR_Plotting',
}

LEGACY_PROCESSORS = {
    'linear_evaluation_att': 'processor.legacy.linear_evaluation_att.LE_Processor',
    'linear_evaluation_with_plotting': 'processor.legacy.linear_evaluation_with_plotting.LE_Processor',
    'pretrain_crossclr': 'processor.legacy.pretrain_crossclr.CrosSCLR_Processor',
    'pretrain_crossclr_3views': 'processor.legacy.pretrain_crossclr_3views.CrosSCLR_3views_Processor',
    'pretrain_skeletonclr_eucl': 'processor.legacy.pretrain_skeletonclr_eucl.SkeletonCLR_Processor',
    'pretrain_skeletonclr_3views_eucl': 'processor.legacy.pretrain_skeletonclr_3views_eucl.SkeletonCLR_3views_Eucl_Processor',
    'pretrain_skeletonclr_att': 'processor.legacy.pretrain_skeletonclr_att.SkeletonCLR_Att_Processor',
    'plot_skeletonclr_eucl': 'processor.legacy.plot_skeletonclr_eucl.SkeletonCLR_Plotting',
    'plot_skeletonclr_mert': 'processor.legacy.plot_skeletonclr_mert.SkeletonCLR_Plotting',
}

PROCESSORS = {**ACTIVE_PROCESSORS, **LEGACY_PROCESSORS}
