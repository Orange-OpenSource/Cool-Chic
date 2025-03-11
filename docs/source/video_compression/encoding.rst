Encoding a video frame
""""""""""""""""""""""

Inter frame encoding have an additional training stage: the pre-training of the
motion decoder to imitate an optical flow obtained by RAFT, an off-the-shelf
optical flow estimator. The duration of this training phase is set by the
``--n_itr_pretrain_motion`` parameter.

See the :ref:`video coding example script <video_coding_example>` for an example
of how videos are encoded using Cool-chic.
