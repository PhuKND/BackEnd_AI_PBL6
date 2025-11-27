package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.requests.CreateUserRequest;
import com.khanhlms.medical_store.dtos.response.UserResponse;
import com.khanhlms.medical_store.dtos.users.requests.ChangeProfileRequest;
import com.khanhlms.medical_store.dtos.users.response.ProfileResponse;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.entities.RoleEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.enums.RoleEnums;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.UserMapper;
import com.khanhlms.medical_store.repositories.RoleRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import com.khanhlms.medical_store.utills.ReflexUtills;
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

        if (createUserRequest.getPassword() == null || createUserRequest.getPassword().isEmpty()
                || createUserRequest.getUsername() == null || createUserRequest.getUsername().isEmpty()
                || createUserRequest.getRepeat() == null || createUserRequest.getRepeat().isEmpty()
        ) {
            throw new AppException(ErrorCode.INVALID_REQUEST);
        }
        if (createUserRequest.getPassword().equals(createUserRequest.getRepeat()) == false) {
            throw new AppException(ErrorCode.PASSWORD_MISMATCH);
        }
        if (createUserRequest.getPassword().length() < 8) {
            throw new AppException(ErrorCode.PASSWORD_TOO_SHORT);
        }
        if (userRepository.findByUsername(createUserRequest.getUsername()).isPresent()) {
            throw new AppException(ErrorCode.USER_EXISTED);
        }
        RoleEntity role = roleRepository.findByName(RoleEnums.USER.name()).get();
        Set<RoleEntity> roleEntities = new HashSet<>();
        roleEntities.add(role);

        UserEntity  userEntity = UserEntity.builder()
                .username(createUserRequest.getUsername())
                .password(passwordEncoder.encode(createUserRequest.getPassword()))
                .isActive(true)
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
    public ProfileResponse handgetMyInfo(String username){
        UserEntity userEntity = this.userRepository.findByUsername(username).get();
        return userMapper.toProfileResponse(userEntity);
    }
    public ProfileResponse handChangeProfile(String username,ChangeProfileRequest changeProfileRequest){
        UserEntity updateUser = userMapper.toUserEntity(changeProfileRequest);
        UserEntity userEntity = this.userRepository.findByUsername(username).get();
        ReflexUtills.mergeNonNullFields(userEntity, updateUser);
        return userMapper.toProfileResponse(userRepository.save(updateUser));
    }
}
