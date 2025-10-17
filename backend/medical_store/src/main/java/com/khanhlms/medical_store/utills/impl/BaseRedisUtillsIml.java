package com.khanhlms.medical_store.utills.impl;

import com.khanhlms.medical_store.utills.BaseRedisUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.HashOperations;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.util.Set;
import java.util.concurrent.TimeUnit;

@Component
public class BaseRedisUtillsIml implements BaseRedisUtils {
    private final RedisTemplate<String, Object> redisTemplate;
    private HashOperations<String, String, Object> hashOperations;

    public BaseRedisUtillsIml(RedisTemplate<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
        this.hashOperations = redisTemplate.opsForHash();
    }


    @Override
    public void set(String key, String value) {
        this.redisTemplate.opsForValue().set(key, value);
    }

    @Override
    public void set(String key, String value, Long expireTime, TimeUnit timeUnit) {
        this.redisTemplate.opsForValue().set(key, value, expireTime, timeUnit);
    }

    @Override
    public void set(String key, Boolean value, Long expireTime, TimeUnit timeUnit) {
        this.redisTemplate.opsForValue().set(key, value, expireTime, timeUnit);
    }

    @Override
    public void hashSet(String key, String field, Object hashValue) {
        this.hashOperations.put(key, field, hashValue);
    }

    @Override
    public void setTimetoLive(String key, Long expireTime, TimeUnit timeUnit) {
        this.redisTemplate.expire(key, expireTime, timeUnit);
    }

    @Override
    public Object getForString(String key) {
        return this.redisTemplate.opsForValue().get(key);
    }

    @Override
    public void deleteForString(String key) {
        this.redisTemplate.delete(key);
    }

    @Override
    public Set<String> getKeys(String pattern) {
        return this.redisTemplate.keys(pattern);
    }
}
