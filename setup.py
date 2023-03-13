# Copyright (c) EqualityAI Corporation.
# Licensed under the Apache License.

import setuptools

long_description = open('PyPI_README.md').read()

# Use requirements.txt to set the install_requires
install_requires = [
    "BlackBoxAuditing",
    "aif360>=0.5.0",
    "dalex>=1.5.0",
    "fairlearn>=0.7.0",
    "pandas>=1.2.5",
    "numpy>=1.20.3",
    "scikit-learn>=0.22.1",
]

extras = {'tests': ["pytest>=7.2.0", "pytest-cov>=2.8.1"], 'doc': ["sphinx"]}

setuptools.setup(
    name="equalityml",
    version="0.1.9-a1",
    author="Ben Brintz, Mark Zhang, James Ng, Janice Davis, Jared Hansen, Ji won Chang, João Granja, Rizwan Muhammad",
    author_email="support@equalityai.com",
    description="Algorithms for evaluating fairness metrics and mitigating unfairness in supervised machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://equalityai.com",
    project_urls={
        "Website": "https://equalityai.com",
        "Manifesto": "https://equalityai.com/community/#manifesto",
        "GitHub": "https://github.com/EqualityAI/EqualityML",
        "Slack": "https://equalityai.slack.com/ssb/redirect#/shared-invite/email"
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    extras_require=extras,
    include_package_data=True,
    zip_safe=False,
)
