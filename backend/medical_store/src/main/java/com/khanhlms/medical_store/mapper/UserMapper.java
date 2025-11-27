package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.response.UserResponse;
import com.khanhlms.medical_store.dtos.users.requests.ChangeProfileRequest;
import com.khanhlms.medical_store.dtos.users.response.ProfileResponse;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.utills.CloudinaryUtils;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.Named;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.multipart.MultipartFile;

@Mapper(componentModel = "spring")
public abstract class UserMapper {
    @Autowired
    protected CloudinaryUtils cloudinaryUtils;

    public abstract UserResponse toResponse(UserEntity userEntity);
    public abstract ProfileResponse toProfileResponse(UserEntity userEntity);
    @Mappings({
            @Mapping(source = "image", target = "avatarUrl", qualifiedByName = "mapImage" )
    })
    public abstract UserEntity toUserEntity(ChangeProfileRequest changeProfileRequest);

    @Named("mapImage")
    protected String mapImage(MultipartFile fileImage) {
        if (fileImage == null || fileImage.isEmpty()) {
            return null;
        }
        return cloudinaryUtils.uploadImageCloddy(fileImage);
    }

}
