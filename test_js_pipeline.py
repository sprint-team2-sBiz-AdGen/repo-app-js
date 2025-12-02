#!/usr/bin/env python3
"""
JS 파트 파이프라인 테스트 스크립트
FastAPI 엔드포인트를 순차적으로 테스트합니다.

사용법:
    python test_js_pipeline.py

필요한 패키지:
    pip install requests pillow
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any

# API 설정
API_BASE_URL = "http://localhost:8012"
# API_BASE_URL = "http://34.9.178.28:8012"  # 원격 서버 사용 시

def print_section(title: str):
    """섹션 구분 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_response(response: requests.Response, step_name: str):
    """응답 출력"""
    print(f"\n[{step_name}]")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return data
    except:
        print(f"Response Text: {response.text}")
        return None

def create_test_image(output_path: str = "/tmp/test_image.jpg"):
    """테스트용 이미지 생성 (PIL 사용)"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 간단한 테스트 이미지 생성
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # 텍스트 추가
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        text = "Test Food Image\n맛있는 음식"
        draw.text((50, 250), text, fill='black', font=font)
        
        img.save(output_path)
        print(f"✅ 테스트 이미지 생성: {output_path}")
        return output_path
    except ImportError:
        print("⚠️  PIL/Pillow가 설치되지 않았습니다. 기존 이미지 파일을 사용하세요.")
        return None

def test_create_job(image_path: str, description: str) -> Optional[str]:
    """1. Job 생성 테스트"""
    print_section("1. Job 생성 (이미지 업로드 + 설명)")
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None
    
    url = f"{API_BASE_URL}/api/v1/jobs/create"
    
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'request': json.dumps({'description': description})
        }
        
        print(f"Request URL: {url}")
        print(f"Description: {description}")
        print(f"Image: {image_path}")
        
        response = requests.post(url, files=files, data=data)
        result = print_response(response, "Job 생성")
        
        if response.status_code == 200 and result:
            job_id = result.get('job_id')
            print(f"✅ Job 생성 성공: {job_id}")
            return job_id
        else:
            print(f"❌ Job 생성 실패")
            return None

def test_kor_to_eng(job_id: str) -> bool:
    """2. 한국어 → 영어 변환 테스트"""
    print_section("2. 한국어 → 영어 변환 (kor-to-eng)")
    
    url = f"{API_BASE_URL}/api/js/gpt/kor-to-eng"
    data = {
        'job_id': job_id,
        'tenant_id': 'test_tenant'
    }
    
    print(f"Request URL: {url}")
    print(f"Job ID: {job_id}")
    
    response = requests.post(url, json=data)
    result = print_response(response, "Kor→Eng 변환")
    
    if response.status_code == 200 and result:
        desc_eng = result.get('desc_eng', '')
        print(f"✅ 변환 성공: {desc_eng}")
        return True
    else:
        print(f"❌ 변환 실패")
        return False

def test_ad_copy_eng(job_id: str) -> bool:
    """3. 영어 광고문구 생성 테스트"""
    print_section("3. 영어 광고문구 생성 (ad-copy-eng)")
    
    url = f"{API_BASE_URL}/api/js/gpt/ad-copy-eng"
    data = {
        'job_id': job_id,
        'tenant_id': 'test_tenant'
    }
    
    print(f"Request URL: {url}")
    print(f"Job ID: {job_id}")
    
    response = requests.post(url, json=data)
    result = print_response(response, "영어 광고문구 생성")
    
    if response.status_code == 200 and result:
        ad_copy_eng = result.get('ad_copy_eng', '')
        print(f"✅ 생성 성공")
        if ad_copy_eng:
            print(f"광고문구: {ad_copy_eng[:100]}...")
        return True
    else:
        print(f"❌ 생성 실패")
        return False

def test_ad_copy_kor(job_id: str) -> bool:
    """4. 한글 광고문구 생성 테스트"""
    print_section("4. 한글 광고문구 생성 (ad-copy-kor)")
    
    url = f"{API_BASE_URL}/api/js/gpt/ad-copy-kor"
    data = {
        'job_id': job_id,
        'tenant_id': 'test_tenant'
    }
    
    print(f"Request URL: {url}")
    print(f"Job ID: {job_id}")
    
    response = requests.post(url, json=data)
    result = print_response(response, "한글 광고문구 생성")
    
    if response.status_code == 200 and result:
        ad_copy_kor = result.get('ad_copy_kor', '')
        print(f"✅ 생성 성공")
        if ad_copy_kor:
            print(f"광고문구: {ad_copy_kor[:100]}...")
        return True
    else:
        print(f"❌ 생성 실패")
        return False

def test_health_check():
    """서버 상태 확인"""
    print_section("0. 서버 상태 확인")
    
    url = f"{API_BASE_URL}/"
    print(f"Request URL: {url}")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ 서버 연결 성공")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 서버에 연결할 수 없습니다: {API_BASE_URL}")
        print(f"   서버가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("  JS 파트 파이프라인 테스트")
    print("=" * 60)
    
    # 서버 상태 확인
    if not test_health_check():
        print("\n❌ 서버 연결 실패. 테스트를 중단합니다.")
        return
    
    # 테스트 이미지 경로 설정
    test_image_path = "/tmp/test_image.jpg"
    if not os.path.exists(test_image_path):
        # 테스트 이미지 생성 시도
        created_path = create_test_image(test_image_path)
        if not created_path:
            # 기존 이미지 파일 사용 (있는 경우)
            possible_paths = [
                "/home/LJS/feedlyai-work/test.jpg",
                "/home/LJS/feedlyai-work/media/uploads/test.jpg",
                "test.jpg"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    test_image_path = path
                    break
            else:
                print(f"\n❌ 테스트 이미지를 찾을 수 없습니다.")
                print(f"   다음 중 하나를 수행하세요:")
                print(f"   1. {test_image_path} 파일 생성")
                print(f"   2. PIL/Pillow 설치: pip install pillow")
                return
    
    # 테스트 설명
    test_description = "맛있는 비빔밥과 김치를 제공하는 한식당입니다. 신선한 재료로 만든 건강한 식사입니다."
    
    # 1. Job 생성
    job_id = test_create_job(test_image_path, test_description)
    if not job_id:
        print("\n❌ Job 생성 실패. 테스트를 중단합니다.")
        return
    
    # 각 단계 사이에 짧은 대기 시간
    time.sleep(1)
    
    # 2. 한국어 → 영어 변환
    if not test_kor_to_eng(job_id):
        print("\n⚠️  Kor→Eng 변환 실패. 다음 단계를 건너뜁니다.")
        return
    
    time.sleep(1)
    
    # 3. 영어 광고문구 생성
    if not test_ad_copy_eng(job_id):
        print("\n⚠️  영어 광고문구 생성 실패. 다음 단계를 건너뜁니다.")
        return
    
    time.sleep(1)
    
    # 4. 한글 광고문구 생성
    if not test_ad_copy_kor(job_id):
        print("\n⚠️  한글 광고문구 생성 실패.")
        return
    
    # 최종 결과
    print_section("테스트 완료")
    print("✅ 모든 파이프라인 단계가 성공적으로 완료되었습니다!")
    print(f"Job ID: {job_id}")

if __name__ == "__main__":
    main()

