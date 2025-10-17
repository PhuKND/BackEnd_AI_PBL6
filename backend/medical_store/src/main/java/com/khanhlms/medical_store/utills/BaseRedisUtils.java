package com.khanhlms.medical_store.utills;

import java.util.Set;
import java.util.concurrent.TimeUnit;

public interface BaseRedisUtils {
    void set(String key, String value);
    void set(String key, String value, Long expireTime,  TimeUnit timeUnit);
    void set(String key, Boolean value, Long expireTime,  TimeUnit timeUnit);

    void hashSet(String key, String field, Object hashValue);

    void setTimetoLive(String key, Long expireTime, TimeUnit timeUnit);
    Object getForString(String key);

    public Set<String> getKeys(String pattern);
    void deleteForString(String key);
}
