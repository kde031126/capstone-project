const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL;

function buildUrl(path: string) {
  if (!API_BASE_URL) {
    throw new Error(
      'Missing EXPO_PUBLIC_API_URL. Add it to your .env file, for example EXPO_PUBLIC_API_URL=http://192.168.1.10:3000'
    );
  }

  const normalizedBaseUrl = API_BASE_URL.replace(/\/$/, '');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  return `${normalizedBaseUrl}${normalizedPath}`;
}

function tryParseJson(text: string) {
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(buildUrl(path), {
    ...options,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
  });

  const rawResponse = await response.text();
  const data = tryParseJson(rawResponse);

  if (!response.ok) {
    const message =
      typeof data === 'string'
        ? data
        : data && typeof data === 'object' && 'message' in data && typeof data.message === 'string'
          ? data.message
          : `Request failed with status ${response.status}`;

    throw new Error(message);
  }

  return data as T;
}

export type ParentLoginPayload = {
  parentEmail: string;
  childName: string;
  age: number;
};

export type ParentLoginResponse = {
  token?: string;
  childId?: string;
  message?: string;
};

const PARENT_LOGIN_ENDPOINT = '/auth/parent-login';

export function submitParentLogin(payload: ParentLoginPayload) {
  return apiFetch<ParentLoginResponse>(PARENT_LOGIN_ENDPOINT, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
