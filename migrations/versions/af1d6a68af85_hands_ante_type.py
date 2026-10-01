"""hands ante_type

Revision ID: af1d6a68af85
Revises: 889149f41a73
Create Date: 2026-10-01 13:31:31.616031

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'af1d6a68af85'
down_revision: Union[str, Sequence[str], None] = '889149f41a73'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Автогенерация не видит CHECK — ограничение добавлено вручную.
    # У уже импортированных раздач тип анте неизвестен (NULL); маппинг считает его как "none".
    with op.batch_alter_table('hands', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ante_type', sa.String(length=8), nullable=True))
        batch_op.create_check_constraint(
            op.f('ck_hands_ante_type'), "ante_type IN ('none', 'each', 'bb')"
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('hands', schema=None) as batch_op:
        batch_op.drop_constraint(op.f('ck_hands_ante_type'), type_='check')
        batch_op.drop_column('ante_type')
