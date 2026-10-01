from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'kf_object_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Kalman Filter object tracker for autonomous systems.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kf_object_tracker        = kf_object_tracking.kf_object_tracker:main',
            'tracker_visualizer       = kf_object_tracking.tracker_visualizer:main',
            'detection2d_visualizer   = kf_object_tracking.detection2d_visualizer:main',
            'dummy_objects_publisher  = kf_object_tracking.dummy_objects_publisher:main',
        ],
    },
)
