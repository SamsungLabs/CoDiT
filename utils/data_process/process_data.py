import os
import sys
from absl import app, flags

sys.path.append(os.getcwd())

FLAGS = flags.FLAGS
flags.DEFINE_string('data_process', 'srrc_dual_frankas', 'data_process id')
flags.DEFINE_string('data_type', 'episodic_data', 'data type')
flags.DEFINE_string(
    'language_embedding', 'binary_encoder', 'method of embedding language description'
)
flags.DEFINE_integer('language_embedding_dim', 512, 'method of embedding language description')
flags.DEFINE_string(
    'source_dir',
    '/home/sr5/dlvr/dataset/raw/srrc_dual_frankas',
    'Source directory of data to process',
)
flags.DEFINE_string(
    'target_dir',
    '/home/sr5/dlvr/dataset/processed/srrc_dual_frankas',
    'Target directory of processed data to be saved',
)
flags.DEFINE_integer('skip_param', 1, 'Skip files parameter')


def main(argv):
    data_process_id = FLAGS.data_process
    data_type = FLAGS.data_type
    source_dir = FLAGS.source_dir
    target_dir = FLAGS.target_dir
    language_embedding = FLAGS.language_embedding
    language_embedding_dim = FLAGS.language_embedding_dim

    if data_type == 'episodic_data':
        import episodic_data_process as data_process
    else:
        raise NotImplementedError

    data_processor = data_process.make(data_process_id)(
        source_dir,
        target_dir,
        language_embedding=language_embedding,
        language_embedding_dim=language_embedding_dim,
    )
    data_processor.process_data()


if __name__ == '__main__':
    app.run(main)
