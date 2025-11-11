import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
    
__version__ = "0.0.0"
    
REPO_NAME = "document_summarization_for_legal_texts"
AUTHOR_USER_NAME = "madhusudhanant"
SRC_REPO = "datascience"
AUTHOR_EMAIL = "mr.vkjilla2024@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for document summarization for legal texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)