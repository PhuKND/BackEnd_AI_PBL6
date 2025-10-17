package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.requests.categories.CreateCategoryRequest;
import com.khanhlms.medical_store.dtos.response.categories.CategoryResponse;
import com.khanhlms.medical_store.entities.CategoryEntity;
import com.khanhlms.medical_store.mapper.CategoriesMapper;
import com.khanhlms.medical_store.repositories.CategoriesRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class CategoriesService {
    CategoriesRepository categoriesRepository;
    CategoriesMapper categoriesMapper;
    public CategoryResponse handCreteCategory(CreateCategoryRequest createCategoryRequest) {
        CategoryEntity categoryEntity = categoriesMapper.toEntity(createCategoryRequest);
        categoryEntity.setPosition((int)this.categoriesRepository.count());
        categoryEntity.setActive(true);
        categoryEntity.setDeleted(false);

        return this.categoriesMapper.toResponse(categoriesRepository.save(categoryEntity));
    }
    public List<CategoryResponse> handfindAll(Pageable pageable) {
        return this.categoriesRepository.findAll(pageable)
                .getContent()
                .stream()
                .map(item -> this.categoriesMapper.toResponse(item))
                .toList();
    }
}
