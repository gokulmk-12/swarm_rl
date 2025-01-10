from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'LLM'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhay',
    maintainer_email='abhay@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'LLM_node = LLM.LLM_node:main',            
        ],
    },
)
