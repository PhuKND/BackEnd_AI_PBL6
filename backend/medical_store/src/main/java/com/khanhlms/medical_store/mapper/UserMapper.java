package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.response.UserResponse;
import com.khanhlms.medical_store.entities.UserEntity;
import org.mapstruct.Mapper;
@Mapper(componentModel = "spring")
public abstract class UserMapper {
    public abstract UserResponse toResponse(UserEntity userEntity);
}
