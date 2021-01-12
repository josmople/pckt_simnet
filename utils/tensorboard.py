def writer(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(
        log_dir=log_dir,
        comment=comment,
        purge_step=purge_step,
        max_queue=max_queue,
        flush_secs=flush_secs,
        filename_suffix=filename_suffix
    )
