from setuptools import find_packages, setup

package_name = 'mine_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/mine_pipeline']),
        ('share/mine_pipeline', ['package.xml']),
        ('share/mine_pipeline/launch', ['launch/pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_sim_node = mine_pipeline.sensor_sim_node:main',
            'preproc_node = mine_pipeline.preproc_node:main',
            'ai_inference_node = mine_pipeline.ai_inference_node:main',
            'alarm_node = mine_pipeline.alarm_node:main',
            'full_pipeline = mine_pipeline.pipeline_launcher:main',
        ],
    },
)