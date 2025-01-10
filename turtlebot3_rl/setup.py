import glob
import os
from setuptools import find_packages, setup

package_name = 'turtlebot3_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', 'turtlebot3_drl_stage3.launch.py'))),
    ],
    install_requires=['setuptools', 'launch'],
    zip_safe=True,
    maintainer='jack',
    maintainer_email='jack@mail.google.ac.in',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'environment = turtlebot3_rl.env:main',
        	'gazebo = turtlebot3_rl.gazebo:main',
        	'train_agent = turtlebot3_rl.agent:main_train',
        	'test_agent = turtlebot3_rl.agent:main_test',
        	'global_database_node = turtlebot3_rl.global_database_node:main',
        	'det_obs = turtlebot3_rl.det_obs:main',
        ],
    },
)
