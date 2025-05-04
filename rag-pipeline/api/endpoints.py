from fastapi import APIRouter, HTTPException
from rag_pipeline.api.models import RAGRequest, RAGResponse
from rag_pipeline.api.services import executar_pipeline_rag

router = APIRouter()

@router.post("/run-rag", response_model=RAGResponse)
async def run_rag(request: RAGRequest):
    """
    Executa o pipeline RAG com os parâmetros fornecidos.
    """
    try:
        resultado = executar_pipeline_rag(
            index_name=request.index_name,
            namespace=request.namespace,
            document_url=request.document_url,
        )
        return RAGResponse(message="Pipeline RAG executado com sucesso!", details=resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
