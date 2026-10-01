"""decision all_in, node_stats action

Revision ID: 889149f41a73
Revises: e8c82a324ca8
Create Date: 2026-10-01 13:14:43.665573

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '889149f41a73'
down_revision: Union[str, Sequence[str], None] = 'e8c82a324ca8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Флаг олл-ина у уже импортированных решений неизвестен (0); чтобы он был верным,
    # раздачи нужно переимпортировать из сырого текста.
    with op.batch_alter_table('decisions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('all_in', sa.Boolean(), server_default=sa.text('0'), nullable=False))

    # Строки отчёта без проверяемого действия бессмысленны; до прогона анализа
    # (тикет 09) таблица не заполнялась.
    op.execute("DELETE FROM node_stats")
    with op.batch_alter_table('node_stats', schema=None) as batch_op:
        batch_op.add_column(sa.Column('action', sa.String(length=16), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('node_stats', schema=None) as batch_op:
        batch_op.drop_column('action')

    with op.batch_alter_table('decisions', schema=None) as batch_op:
        batch_op.drop_column('all_in')
