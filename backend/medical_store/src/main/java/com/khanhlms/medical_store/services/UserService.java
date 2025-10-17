package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.requests.CreateUserRequest;
import com.khanhlms.medical_store.dtos.response.UserResponse;
import com.khanhlms.medical_store.entities.RoleEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.enums.RoleEnums;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.UserMapper;
import com.khanhlms.medical_store.repositories.RoleRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class UserService {
    RoleRepository roleRepository;
    UserRepository userRepository;
    PasswordEncoder passwordEncoder;
    UserMapper userMapper;
    public boolean handlCreateUser(CreateUserRequest createUserRequest) {
        if (userRepository.findByUsername(createUserRequest.getUsername()).isPresent()) {
            throw new AppException(ErrorCode.USER_EXISTED);
        }
        RoleEntity role = roleRepository.findByName(RoleEnums.USER.name()).get();
        Set<RoleEntity> roleEntities = new HashSet<>();
        roleEntities.add(role);

        UserEntity  userEntity = UserEntity.builder()
                .username(createUserRequest.getUsername())
                .password(passwordEncoder.encode(createUserRequest.getPassword()))
                .roles(roleEntities)
                .build();
        userRepository.save(userEntity);
        return true;
    }
    public List<UserResponse> handleUserList(){
        List<UserResponse> userResponseList = this.userRepository.findAll()
                .stream()
                .map(user -> userMapper.toResponse(user))
                .toList();
        return userResponseList;
    }
}
