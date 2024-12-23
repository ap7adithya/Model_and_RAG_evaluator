# app/api/endpoints/rag_evaluator.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
import tempfile
from pathlib import Path
from knowledge_base_fetcher import fetch_knowledge_bases, get_knowledge_base
from orchestrator import final_rag_evaluator
import re
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/knowledge-bases")
async def get_knowledge_bases():
    try:
        all_knowledge_bases = []
        for kb in fetch_knowledge_bases():
            kb_details = get_knowledge_base(kb['knowledgeBaseId'])
            all_knowledge_bases.append({
                'name': kb_details['name'],
                'id': kb_details['knowledgeBaseId'],
                'embedding_model_arn': kb_details['knowledgeBaseConfiguration']['vectorKnowledgeBaseConfiguration']['embeddingModelArn']
            })
        return all_knowledge_bases
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_rag(
    questions_file: UploadFile = File(...),
    answers_file: UploadFile = File(...),
    knowledge_base_ids: List[str] = Form(...)
):
    try:
        # Validate file types
        if not questions_file.filename.endswith('.csv') or not answers_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Process knowledge base IDs - handle both comma-separated and array formats
        kb_ids = []
        for kb_id in knowledge_base_ids:
            # Split if comma-separated
            if ',' in kb_id:
                kb_ids.extend([id.strip() for id in kb_id.split(',')])
            else:
                kb_ids.append(kb_id.strip())

        # Remove duplicates and empty strings
        kb_ids = list(filter(None, set(kb_ids)))

        # Ensure we have at least two knowledge bases for comparison
        if len(kb_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="Please provide at least two knowledge base IDs for comparison"
            )

        # Validate knowledge base IDs
        for kb_id in kb_ids:
            if not re.match(r'^[0-9a-zA-Z]{10}$', kb_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid knowledge base ID: {kb_id}. ID must be exactly 10 alphanumeric characters."
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            questions_path = Path(temp_dir) / questions_file.filename
            answers_path = Path(temp_dir) / answers_file.filename
            
            with open(questions_path, "wb") as f:
                f.write(await questions_file.read())
            with open(answers_path, "wb") as f:
                f.write(await answers_file.read())

            # Get knowledge base details
            selected_kbs = []
            for kb_id in kb_ids:
                try:
                    kb_details = get_knowledge_base(kb_id)
                    selected_kbs.append({
                        'name': kb_details['name'],
                        'id': kb_details['knowledgeBaseId'],
                        'embedding_model_arn': kb_details['knowledgeBaseConfiguration']['vectorKnowledgeBaseConfiguration']['embeddingModelArn']
                    })
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error retrieving knowledge base {kb_id}: {str(e)}"
                    )

            # Run evaluation
            results_df, evaluation_results, costs_eval_results, score_rubric_df = final_rag_evaluator(
                questions_path, 
                answers_path,
                selected_kbs
            )

            return {
                "results": results_df.to_dict(orient='records'),
                "evaluation_summary": evaluation_results,
                "cost_analysis": costs_eval_results,
                "scoring_rubric": score_rubric_df.to_dict(orient='records')
            }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))