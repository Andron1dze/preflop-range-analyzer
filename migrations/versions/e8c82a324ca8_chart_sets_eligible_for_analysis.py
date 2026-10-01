"""chart_sets eligible_for_analysis

Revision ID: e8c82a324ca8
Revises: eac2cfd810ac
Create Date: 2026-10-01 12:50:44.355610

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8c82a324ca8'
down_revision: Union[str, Sequence[str], None] = 'eac2cfd810ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Уже загруженные наборы получают флаг по источнику, затем колонка становится NOT NULL.
    with op.batch_alter_table('chart_sets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('eligible_for_analysis', sa.Boolean(), nullable=True))
    op.execute(
        "UPDATE chart_sets SET eligible_for_analysis = (source != 'synthetic-test')"
    )
    with op.batch_alter_table('chart_sets', schema=None) as batch_op:
        batch_op.alter_column('eligible_for_analysis', existing_type=sa.Boolean(), nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('chart_sets', schema=None) as batch_op:
        batch_op.drop_column('eligible_for_analysis')
