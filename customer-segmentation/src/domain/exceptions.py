# -*- coding: utf-8 -*-


# Classe base para todos os erros de dominio do projeto de segmentacao
class DomainError(Exception): ...

# Lancada quando os dados de entrada falham na validacao de schema ou regras de negocio
class ValidationError(DomainError): ...

# Lancada quando tentativa de inferencia ocorre antes do modelo ser carregado
class ModelNotLoadedError(DomainError): ...

# Lancada quando ocorre erro durante o processo de segmentacao
class SegmentationError(DomainError): ...

# Lancada quando ocorre erro ao carregar ou salvar artefatos do modelo
class RepositoryError(DomainError): ...
