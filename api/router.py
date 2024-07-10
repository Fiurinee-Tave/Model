import uvicorn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .flower_recommendation_re import recommender

#라우터 객체 생성
router = APIRouter() #여러 엔드포인트(API 경로)를 그룹화하고 관리하는 데 사용

class RecommendRequest(BaseModel):
    user_input: str
    user_month: int

@router.post('/recommend')
def get_recommendations(request: RecommendRequest):
    try:
        recommendations = recommender.recommend(request.user_input, request.user_month)
        # result = recommendations.to_dict(orient='records')
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))