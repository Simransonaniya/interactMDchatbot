"""Initial schema migration for InteractMD.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-09-22 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('first_name', sa.String(length=100), nullable=False),
        sa.Column('last_name', sa.String(length=100), nullable=False),
        sa.Column('role', sa.String(length=20), server_default='LEARNER', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # 2. Cases table
    op.create_table(
        'cases',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('specialty', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('difficulty', sa.String(length=20), server_default='Beginner', nullable=False),
        sa.Column('is_published', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cases_is_published'), 'cases', ['is_published'], unique=False)

    # 3. Patient Profiles table
    op.create_table(
        'patient_profiles',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('case_id', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('age', sa.Integer(), nullable=False),
        sa.Column('gender', sa.String(length=20), nullable=False),
        sa.Column('occupation', sa.String(length=150), nullable=True),
        sa.Column('persona', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('case_id')
    )
    op.create_index(op.f('ix_patient_profiles_case_id'), 'patient_profiles', ['case_id'], unique=True)

    # 4. Clinical Facts table
    op.create_table(
        'clinical_facts',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('case_id', sa.String(length=50), nullable=False),
        sa.Column('chief_complaint', sa.Text(), nullable=True),
        sa.Column('history', sa.Text(), nullable=True),
        sa.Column('onset', sa.Text(), nullable=True),
        sa.Column('timing', sa.Text(), nullable=True),
        sa.Column('location', sa.Text(), nullable=True),
        sa.Column('character', sa.Text(), nullable=True),
        sa.Column('severity', sa.Text(), nullable=True),
        sa.Column('radiation', sa.Text(), nullable=True),
        sa.Column('aggravating_factors', sa.Text(), nullable=True),
        sa.Column('relieving_factors', sa.Text(), nullable=True),
        sa.Column('associated_symptoms', sa.JSON(), nullable=True),
        sa.Column('past_medical_history', sa.JSON(), nullable=True),
        sa.Column('medications', sa.JSON(), nullable=True),
        sa.Column('allergies', sa.JSON(), nullable=True),
        sa.Column('family_history', sa.Text(), nullable=True),
        sa.Column('social_history', sa.Text(), nullable=True),
        sa.Column('smoking', sa.Text(), nullable=True),
        sa.Column('alcohol', sa.Text(), nullable=True),
        sa.Column('facts_json', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('case_id')
    )
    op.create_index(op.f('ix_clinical_facts_case_id'), 'clinical_facts', ['case_id'], unique=True)

    # 5. Physical Findings table
    op.create_table(
        'physical_findings',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('case_id', sa.String(length=50), nullable=False),
        sa.Column('system', sa.String(length=50), nullable=False),
        sa.Column('finding', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_physical_findings_case_id'), 'physical_findings', ['case_id'], unique=False)

    # 6. Investigations table
    op.create_table(
        'investigations',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('case_id', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('result', sa.Text(), nullable=False),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('reference_range', sa.String(length=100), nullable=True),
        sa.Column('is_available', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_investigations_case_id'), 'investigations', ['case_id'], unique=False)

    # 7. Case Hidden Evaluations table (sensitive)
    op.create_table(
        'case_hidden_evaluations',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('case_id', sa.String(length=50), nullable=False),
        sa.Column('diagnosis', sa.String(length=255), nullable=False),
        sa.Column('differential_diagnosis', sa.JSON(), nullable=True),
        sa.Column('management', sa.JSON(), nullable=True),
        sa.Column('learning_objectives', sa.JSON(), nullable=True),
        sa.Column('scoring_rubric', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('case_id')
    )
    op.create_index(op.f('ix_case_hidden_evaluations_case_id'), 'case_hidden_evaluations', ['case_id'], unique=True)

    # 8. Simulation Sessions table
    op.create_table(
        'simulation_sessions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('case_id', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='ACTIVE', nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_simulation_sessions_case_id'), 'simulation_sessions', ['case_id'], unique=False)
    op.create_index(op.f('ix_simulation_sessions_status'), 'simulation_sessions', ['status'], unique=False)
    op.create_index(op.f('ix_simulation_sessions_user_id'), 'simulation_sessions', ['user_id'], unique=False)

    # 9. Messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('sender', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['simulation_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_created_at'), 'messages', ['created_at'], unique=False)
    op.create_index(op.f('ix_messages_session_id'), 'messages', ['session_id'], unique=False)

    # 10. Evaluations table
    op.create_table(
        'evaluations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('score', sa.Integer(), server_default='0', nullable=False),
        sa.Column('feedback', sa.Text(), nullable=False),
        sa.Column('strengths', sa.JSON(), nullable=True),
        sa.Column('areas_for_improvement', sa.JSON(), nullable=True),
        sa.Column('category_scores', sa.JSON(), nullable=True),
        sa.Column('detailed_rubric', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['simulation_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_evaluations_session_id'), 'evaluations', ['session_id'], unique=True)
    op.create_index(op.f('ix_evaluations_user_id'), 'evaluations', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('evaluations')
    op.drop_table('messages')
    op.drop_table('simulation_sessions')
    op.drop_table('case_hidden_evaluations')
    op.drop_table('investigations')
    op.drop_table('physical_findings')
    op.drop_table('clinical_facts')
    op.drop_table('patient_profiles')
    op.drop_table('cases')
    op.drop_table('users')
