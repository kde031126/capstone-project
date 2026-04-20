// 로그인 등 사용자 정보 관리

// frontend/api/auth.ts

import { apiFetch } from './api';

export interface UserCreatePayload {
  firebase_uid?: string; 
  user_name: string;
  guardian_name: string;
  birth_year: number;
}

export interface UserResponse {
  user_id: number;
  firebase_uid: string;
  user_name: string;
  guardian_name: string;
  birth_year: number;
  created_at: string;
}

// BE의 router prefix가 /users라면 주소를 맞춰줍니다.
export const loginOrCreateUser = (payload: UserCreatePayload) => {
  return apiFetch<UserResponse>('/users/', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
};