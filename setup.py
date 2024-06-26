from distutils.core import setup
setup(
  name = 'inference3d',
  packages = ['inference3d'],
  version = '0.0.1',
  description = 'Infers 3d path from observation vectors at different times.',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/SheffieldMLtracking/inference3d.git',
  download_url = 'https://github.com/SheffieldMLtracking/inference3d.git',
  keywords = ['flight','bayesian','inference','3d','position','orientation'],
  classifiers = [],
  install_requires=['numpy']
)
