# -*- coding: utf-8 -*-
import os
import sys

from invoke import task

docs_dir = 'docs'
build_dir = os.path.join(docs_dir, '_build')

@task
def test(ctx):
    flake(ctx)
    import pytest
    errcode = pytest.main(['tests'])
    sys.exit(errcode)

@task
def flake(ctx):
    """Run flake8 on codebase."""
    ctx.run('flake8 .', echo=True)

@task
def watch(ctx):
    """Run tests when a file changes. Requires pytest-xdist."""
    import pytest
    errcode = pytest.main(['-f'])
    sys.exit(errcode)

@task
def clean(ctx):
    ctx.run('rm -rf build')
    ctx.run('rm -rf dist')
    ctx.run('rm -rf doc_scanner.egg-info')
    clean_docs(ctx)
    print('Cleaned up.')

@task
def clean_docs(ctx):
    ctx.run('rm -rf %s' % build_dir, echo=True)
