import { createClient } from '@supabase/supabase-js';
import { env } from './env.js';

export const supabaseClient = createClient(
  env.SUPABASE_URL || '',
  env.SUPABASE_ANON_KEY || ''
);