'''
climtools ...
'''

classifiers = '''
              Development Status :: beta
              Environment :: Console
              Intended Audience :: Science/Research
              Intended Audience :: Developers
              License :: GNU GENERAL PUBLIC LICENSE
              Operating System :: OS Independent
              Programming Language :: Python
              Topic :: Scientific/Engineering
              Topic :: Software Development :: Libraries :: Python Modules
              '''

from numpy.distutils.core import Extension
from numpy.distutils.command.install import install
from glob import glob


class my_install(install):
    def run(self):
        install.run(self)

        print '''
        enjoy climtools
        '''


doclines = __doc__.split("\n")

if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(name='climtools',
          version=0,
          description=doclines[0],
          long_description="\n".join(doclines[2:]),
          author='Joao Teixeira',
          author_email='jcmt87@gmail.com',
          url='NA',
          packages=['.'],
          license='GNU',
          platforms=['any'],
          ext_modules=[],
          classifiers=filter(None, classifiers.split('\n')),
          cmdclass={'install': my_install},
          )
