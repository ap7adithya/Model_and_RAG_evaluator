# # # app/api/endpoints/model_evaluator.py
# # from fastapi import APIRouter, File, UploadFile, Form, HTTPException
# # from typing import List
# # import tempfile
# # from pathlib import Path
# # from orchestrator import final_evaluator
# # from PyPDF2 import PdfReader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter

# # router = APIRouter()

# # def chunk_text(text, max_tokens=3800):
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=max_tokens,
# #         chunk_overlap=200,
# #         length_function=len,
# #         separators=["\n\n", "\n", " ", ""]
# #     )
# #     chunks = text_splitter.split_text(text)
# #     return chunks[0] if chunks else text

# # @router.post("/evaluate")
# # async def evaluate_models(
# #     document: UploadFile = File(...),
# #     model_ids: List[str] = Form(...),
# #     summary_prompt: str = Form("Summarize this document in 2 sentences."),
# #     max_tokens: int = Form(4096)
# # ):
# #     try:
# #         # Validate file type
# #         if not document.filename.endswith('.pdf'):
# #             raise HTTPException(status_code=400, detail="Only PDF files are supported")

# #         with tempfile.TemporaryDirectory() as temp_dir:
# #             # Save uploaded PDF
# #             doc_path = Path(temp_dir) / document.filename
# #             with open(doc_path, "wb") as f:
# #                 f.write(await document.read())

# #             # Extract and chunk text
# #             reader = PdfReader(doc_path)
# #             text = ""
# #             for page in reader.pages:
# #                 text += page.extract_text()
# #             chunked_text = chunk_text(text)

# #             # Run evaluation
# #             results_df, evaluation_results, costs_eval_results, score_rubric_df = final_evaluator(
# #                 doc_path,
# #                 model_ids,
# #                 summary_prompt,
# #                 max_tokens,
# #                 text=chunked_text
# #             )

# #             return {
# #                 "results": results_df.to_dict(orient='records'),
# #                 "evaluation_summary": evaluation_results,
# #                 "cost_analysis": costs_eval_results,
# #                 "scoring_rubric": score_rubric_df.to_dict(orient='records')
# #             }

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))


# # app/api/endpoints/model_evaluator.py
# from fastapi import APIRouter, File, UploadFile, Form, HTTPException
# from typing import List
# import tempfile
# from pathlib import Path
# from orchestrator import final_evaluator
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# router = APIRouter()

# def chunk_text(text, max_tokens=3800):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=max_tokens,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks[0] if chunks else text

# @router.post("/evaluate")
# async def evaluate_models(
#     document: UploadFile = File(...),
#     model_ids: List[str] = Form(...),
#     summary_prompt: str = Form("Summarize this document in 2 sentences."),
#     max_tokens: int = Form(4096)
# ):
#     try:
#         # Validate file type
#         if not document.filename.endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="Only PDF files are supported")

#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Save uploaded PDF
#             doc_path = Path(temp_dir) / document.filename
#             with open(doc_path, "wb") as f:
#                 f.write(await document.read())

#             # Process model IDs list if needed
#             processed_model_ids = []
#             for model_id in model_ids:
#                 if ',' in model_id:
#                     processed_model_ids.extend([id.strip() for id in model_id.split(',')])
#                 else:
#                     processed_model_ids.append(model_id.strip())

#             # Run evaluation without the text parameter
#             results_df, evaluation_results, costs_eval_results, score_rubric_df = final_evaluator(
#                 doc_path,
#                 processed_model_ids,
#                 summary_prompt,
#                 max_tokens
#             )

#             return {
#                 "results": results_df.to_dict(orient='records'),
#                 "evaluation_summary": evaluation_results,
#                 "cost_analysis": costs_eval_results,
#                 "scoring_rubric": score_rubric_df.to_dict(orient='records')
#             }

#     except Exception as e:
#         if isinstance(e, HTTPException):
#             raise e
#         raise HTTPException(status_code=500, detail=str(e))

# # app/api/endpoints/model_evaluator.py
# from fastapi import APIRouter, File, UploadFile, Form, HTTPException
# from typing import List
# import tempfile
# from pathlib import Path
# from orchestrator import final_evaluator
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# router = APIRouter()

# def chunk_text(text, max_tokens=3800):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=max_tokens,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks[0] if chunks else text

# @router.post("/evaluate")
# async def evaluate_models(
#     document: UploadFile = File(...),
#     model_ids: List[str] = Form(...),
#     summary_prompt: str = Form("Summarize this document in 2 sentences."),
#     max_tokens: int = Form(4096)
# ):
#     try:
#         # Validate file type
#         if not document.filename.endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="Only PDF files are supported")

#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Save uploaded PDF
#             doc_path = Path(temp_dir) / document.filename
#             with open(doc_path, "wb") as f:
#                 f.write(await document.read())

#             # Process model IDs list if needed
#             processed_model_ids = []
#             for model_id in model_ids:
#                 if ',' in model_id:
#                     processed_model_ids.extend([id.strip() for id in model_id.split(',')])
#                 else:
#                     processed_model_ids.append(model_id.strip())

#             # Run evaluation
#             results_df, evaluation_results, costs_eval_results, score_rubric_df = final_evaluator(
#                 doc_path,
#                 processed_model_ids,
#                 summary_prompt,
#                 max_tokens
#             )

#             # Debug prints
#             print("\nDebug Information:")
#             print("DataFrame Columns:", results_df.columns.tolist())

#             try:
#                 # Convert results to dict with proper column handling
#                 results_data = []
#                 for _, row in results_df.iterrows():
#                     # Create dictionary with all available columns
#                     row_dict = {
#                         'Model': row['Model'],
#                         'Time Length': row['Time Length'],
#                         'Character Count': row['Character Count'],
#                         'Char Process Time': row['Char Process Time'],
#                         'Input Cost': row['Input Cost'],
#                         'Output Cost': row['Output Cost'],
#                         'Total Cost': row['Total Cost(1000)'] if 'Total Cost(1000)' in row else row['Total Cost'],
#                         'Summary Score': row['Summary Score'],
#                         'Invoke Response': row['Invoke Response']
#                     }
#                     results_data.append(row_dict)

#                 # Convert scoring rubric
#                 scoring_data = score_rubric_df.to_dict(orient='records') if not score_rubric_df.empty else []

#                 return {
#                     "status": "success",
#                     "data": {
#                         "results": results_data,
#                         "evaluation_summary": evaluation_results,
#                         "cost_analysis": costs_eval_results,
#                         "scoring_rubric": scoring_data
#                     },
#                     "model_comparison": {
#                         "models_evaluated": processed_model_ids,
#                         "document_name": document.filename,
#                         "summary_prompt": summary_prompt
#                     }
#                 }

#             except Exception as e:
#                 print(f"Error processing results: {str(e)}")
#                 # Print more debug information
#                 print("DataFrame head:")
#                 print(results_df.head())
#                 print("\nDataFrame info:")
#                 print(results_df.info())
#                 raise HTTPException(
#                     status_code=500,
#                     detail=f"Error processing results: {str(e)}"
#                 )

#     except Exception as e:
#         print(f"Full error details: {str(e)}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Error processing request: {str(e)}\nType: {type(e).__name__}"
#         )

#     finally:
#         if 'doc_path' in locals() and doc_path.exists():
#             try:
#                 doc_path.unlink()
#             except Exception:
#                 pass


# app/api/endpoints/model_evaluator.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
import tempfile
from pathlib import Path
from orchestrator import final_evaluator
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import asyncio

router = APIRouter()

def chunk_text(text, max_tokens=3800):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks[0] if chunks else text

@router.post("/evaluate")
async def evaluate_models(
    document: UploadFile = File(...),
    model_ids: List[str] = Form(...),
    summary_prompt: str = Form("Summarize this document in 2 sentences."),
    max_tokens: int = Form(4096)
):
    try:
        # Validate file type
        if not document.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded PDF
            doc_path = Path(temp_dir) / document.filename
            with open(doc_path, "wb") as f:
                f.write(await document.read())

            # Process model IDs list if needed
            processed_model_ids = []
            for model_id in model_ids:
                if ',' in model_id:
                    processed_model_ids.extend([id.strip() for id in model_id.split(',')])
                else:
                    processed_model_ids.append(model_id.strip())

            # Create empty DataFrame with expected columns
            empty_df = pd.DataFrame(columns=[
                'Model', 'Time Length', 'Character Count', 'Char Process Time',
                'Input Cost', 'Output Cost', 'Total Cost', 'Total Cost(1000)',
                'Summary Score', 'Invoke Response'
            ])

            try:
                # Run evaluation in background
                loop = asyncio.get_event_loop()
                results_df, evaluation_results, costs_eval_results, score_rubric_df = await loop.run_in_executor(
                    None,
                    final_evaluator,
                    doc_path,
                    processed_model_ids,
                    summary_prompt,
                    max_tokens
                )

                # Ensure we have a DataFrame even if empty
                results_df = pd.concat([empty_df, results_df]) if not results_df.empty else empty_df

                # Debug prints
                print("\nDebug Information:")
                print("DataFrame Columns:", results_df.columns.tolist())
                print("DataFrame head:", results_df.head())

                # Convert results to dict safely
                results_data = []
                for _, row in results_df.iterrows():
                    try:
                        row_dict = {
                            'Model': row.get('Model', ''),
                            'Time Length': row.get('Time Length', 0),
                            'Character Count': row.get('Character Count', 0),
                            'Char Process Time': row.get('Char Process Time', 0),
                            'Input Cost': row.get('Input Cost', 0),
                            'Output Cost': row.get('Output Cost', 0),
                            'Total Cost': row.get('Total Cost(1000)', row.get('Total Cost', 0)),
                            'Summary Score': row.get('Summary Score', 0),
                            'Invoke Response': row.get('Invoke Response', '')
                        }
                        results_data.append(row_dict)
                    except Exception as row_error:
                        print(f"Error processing row: {str(row_error)}")
                        continue

                return {
                    "status": "success",
                    "data": {
                        "results": results_data,
                        "evaluation_summary": evaluation_results,
                        "cost_analysis": costs_eval_results,
                        "scoring_rubric": score_rubric_df.to_dict(orient='records') if not score_rubric_df.empty else []
                    },
                    "model_comparison": {
                        "models_evaluated": processed_model_ids,
                        "document_name": document.filename,
                        "summary_prompt": summary_prompt
                    }
                }

            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error in evaluation: {str(e)}"
                )

    except Exception as e:
        print(f"Full error details: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}\nType: {type(e).__name__}"
        )

    finally:
        if 'doc_path' in locals() and doc_path.exists():
            try:
                doc_path.unlink()
            except Exception:
                pass