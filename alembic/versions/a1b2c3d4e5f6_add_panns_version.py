"""add panns_version to tracks

Revision ID: a1b2c3d4e5f6
Revises: 1e603a044161
Create Date: 2026-04-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '1e603a044161'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('tracks') as batch_op:
        batch_op.add_column(sa.Column('panns_version', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('tracks') as batch_op:
        batch_op.drop_column('panns_version')
