"""
WRITEME
"""

from sqlalchemy import Column, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


# Embedding Table
class Embedding(Base):
    __tablename__ = "embedding"

    hashid = Column(String(128), primary_key=True)
    numpy_vector = Column(LargeBinary)


"""
def create_db_and_table(embedding_model_name):  
    engine = create_engine(f'sqlite:////full/path/to/{embedding_model_name}.db')
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)

embedding_model_name="your_model_name"
db_session = create_db_and_table(embedding_model_name)
session = db_session()

new_embedding = Embedding(sha512_hash='sample_hash', numpy_vector=b'sample_numpy_vector')
session.add(new_embedding)

session.commit()
"""
