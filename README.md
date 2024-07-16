# Model
## 💻 Technology
***
![python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)&nbsp; ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) &nbsp;<br>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) &nbsp; ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) &nbsp; ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) &nbsp; ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)&nbsp;<br>
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)&nbsp; ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)&nbsp; ![Nginx](https://img.shields.io/badge/nginx-%23009639.svg?style=for-the-badge&logo=nginx&logoColor=white) &nbsp; ![amazon](https://img.shields.io/badge/Amazon_AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white) 
<br>
<br>

## 🧑🏻‍💻Team Member
|**김민지**|**원종효**|**김헤원**|
|:-----:|:-----:|:-----:|
| [<img src="https://github.com/Tave-13th-Project-Team-4-Fiurinee/.github/blob/main/profile/image/%EA%B9%80%EB%AF%BC%EC%A7%80.jpg" alt="김민지" width="150" height="150"> <br/> @ggororok](https://github.com/ggororok) | [<img src="https://github.com/Tave-13th-Project-Team-4-Fiurinee/.github/blob/main/profile/image/%EC%9B%90%EC%A2%85%ED%9A%A8.jpg" alt="원종효" width="150" height="150"> <br/> wonjonghyo](https://github.com/wonjonghyo) | [<img src="https://github.com/user-attachments/assets/e30a7fea-26a0-4b2b-bbef-6706085fb2b7" alt="김혜원" width="150" height="150"> <br/> @hxe0ne](https://github.com/hxe0ne) |
<br>

## 📅 개발 기간
***
 - 2024-05-12 ~ 2024-07-16
<br>

## 🌷 모델 구현 과정
***

### 1️⃣ 사용자 입력과 데이터셋을 이용하 가장 어울리는 꽃을 제공해주는 추천 시스템 구현 (3개의 꽃 추천)
- 데이터 구축: 최종 837개의 데이터 사용 </br>
</tab> - "농촌진흥청 국립원예특작과학원 오늘의 꽃"(open API), "순천만국가정원 탄생화"(pdf), “어니스트 플라워”의 꽃도감(웹크롤링/Selenium) 데이터 사용 </br>
</tab> - 꽃 이름, 월, 일자, 계절, 꽃말, 설명(기념일 등 따로 추가) , 이미지(링크), 색상, (선택한 색상, 중앙 색상들)  칼럼 사용 </br>
</tab> - 월을 통해 계절 추출 </br>
</tab> - 순천만국가정원 데이터 기준으로 일자 채우기, 없는 일자는 월별 탄생화로 채움 </br>
</tab> - 상황(기념일) 검색해서 설명 칼럼에 추가 </br>
</tab> - 나무 데이터 삭제, 이미지 링크 변경 </br>
</tab> - 완전히 중복되는 데이터 합치기 (중복제거) </br>
</tab> - 사용자 입력에 월이 나오지 않는 경우, 사용자 로그를 통해 월,계절 가져오기 </br>

- 색상 추출 </br>
</tab> - PIL, 군집화를 통해 이미지에서 4개의 중심 색상 추출하여 가장 적합한 1개의 색상 선정 </br>
</tab> - HSV코드를 이용한 8개의 색상 군집화를 통해 색상 칼럼 생성

- 모델링(bert) </br>
</tab> 1. 전처리 </br>
</tab> 2. 토큰화, 벡터화(bert) </br>
</tab> 3. 사용자 입력 및 유사도 계산 </br>

- 성능평가 (벡터:bert, bigbird, KLUEroberta / 유사도: cosine, vec db) </br>
</tab> 1. 테스트셋 구축 (월, 계절, 색상 키워드 기준) </br>
</tab> 2. 키워드 유무 기준으로 성능 평가 </br>
</tab> *참고용 정도로만 사용
</tab> </tab> - 간단 비교 결과 bert(cosine), KLUE(cosine)으로 결정 -> 정확도 비교 결과 KLUE(cosine)으로 결정


### 2️⃣ 생성형 모델을 통해 선정 꽃과 어울리는 선물 멘트 생성
- openAI의 API를 가져와서 멘트 생성 (GPT-4) 이용

### 3️⃣ 선정된 꽃과 어울리는 색상의 꽃들 추천(2개의 꽃 추천) 
- 컬러 팔레트 사용: Color Hunt 사이트  https://colorhunt.co/ </br>
</tab> - Selenium을 통한 웹크롤링을 기반으로 컬러 팔레트의 rgb 코드 가져오기 (인기순, 찜 1000개 기준) </br>
</tab> - 선정된 꽃의 RGB의 값과 가장 유사한 색상을 가진 팔레트 추출 (유클리드 거리 이용) </br>
</tab> - 선정된 팔레트에서 어두운 색상 제거, 선정된 색(본인의 색) 제외한 팔레트 색상으로 어울리는 색상 꽃1, 꽃2 추출 (유클리드 거리 이용)

### 4️⃣ 서버 업로드 
- FastAPI를 이용하여 API 키 생성 </br>
- AWS의 EC2를 통해서 배포 (nginx, ubunta 활용)
