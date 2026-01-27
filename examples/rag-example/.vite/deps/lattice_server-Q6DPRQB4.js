import "./chunk-V6TY7KAL.js";

// node_modules/lattice-db/wasm/lattice_server.js
var LatticeDB = class {
  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    LatticeDBFinalization.unregister(this);
    return ptr;
  }
  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_latticedb_free(ptr, 0);
  }
  /**
   * Add an edge between two points
   *
   * # Arguments
   * * `collection` - Collection name
   * * `from_id` - Source point ID
   * * `to_id` - Target point ID
   * * `relation` - Relation type name
   * * `weight` - Optional edge weight (default: 1.0)
   * @param {string} collection
   * @param {bigint} from_id
   * @param {bigint} to_id
   * @param {string} relation
   * @param {number | null} [weight]
   */
  addEdge(collection, from_id, to_id, relation, weight) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(relation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_addEdge(this.__wbg_ptr, ptr0, len0, from_id, to_id, ptr1, len1, isLikeNone(weight) ? 4294967297 : Math.fround(weight));
    if (ret[1]) {
      throw takeFromExternrefTable0(ret[0]);
    }
  }
  /**
   * Create a new collection
   *
   * # Arguments
   * * `name` - Collection name
   * * `config` - Configuration object: `{ vectors: { size: number, distance: string }, hnsw_config?: {...} }`
   *
   * # Returns
   * Result object with status
   * @param {string} name
   * @param {any} config
   * @returns {any}
   */
  createCollection(name, config) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_createCollection(this.__wbg_ptr, ptr0, len0, config);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Delete a collection
   *
   * # Arguments
   * * `name` - Collection name to delete
   *
   * # Returns
   * Boolean indicating success
   * @param {string} name
   * @returns {boolean}
   */
  deleteCollection(name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_deleteCollection(this.__wbg_ptr, ptr0, len0);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] !== 0;
  }
  /**
   * Delete points by IDs
   *
   * # Arguments
   * * `collection` - Collection name
   * * `ids` - Array of point IDs to delete
   *
   * # Returns
   * Number of points deleted
   * @param {string} collection
   * @param {BigUint64Array} ids
   * @returns {any}
   */
  deletePoints(collection, ids) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray64ToWasm0(ids, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_deletePoints(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Get collection info
   *
   * # Arguments
   * * `name` - Collection name
   *
   * # Returns
   * Collection info object
   * @param {string} name
   * @returns {any}
   */
  getCollection(name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_getCollection(this.__wbg_ptr, ptr0, len0);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Get points by IDs
   *
   * # Arguments
   * * `collection` - Collection name
   * * `ids` - Array of point IDs
   * * `with_payload` - Include payload in results (default: true)
   * * `with_vector` - Include vector in results (default: false)
   *
   * # Returns
   * Array of point records
   * @param {string} collection
   * @param {BigUint64Array} ids
   * @param {boolean | null} [with_payload]
   * @param {boolean | null} [with_vector]
   * @returns {any}
   */
  getPoints(collection, ids, with_payload, with_vector) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray64ToWasm0(ids, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_getPoints(this.__wbg_ptr, ptr0, len0, ptr1, len1, isLikeNone(with_payload) ? 16777215 : with_payload ? 1 : 0, isLikeNone(with_vector) ? 16777215 : with_vector ? 1 : 0);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * List all collections
   *
   * # Returns
   * Array of collection names
   * @returns {any}
   */
  listCollections() {
    const ret = wasm.latticedb_listCollections(this.__wbg_ptr);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Create a new LatticeDB instance
   */
  constructor() {
    const ret = wasm.latticedb_new();
    this.__wbg_ptr = ret >>> 0;
    LatticeDBFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
   * Execute a Cypher query
   *
   * # Arguments
   * * `collection` - Collection name
   * * `cypher` - Cypher query string
   * * `parameters` - Optional query parameters
   *
   * # Returns
   * Query result with columns and rows
   * @param {string} collection
   * @param {string} cypher
   * @param {any | null} [parameters]
   * @returns {any}
   */
  query(collection, cypher, parameters) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(cypher, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_query(this.__wbg_ptr, ptr0, len0, ptr1, len1, isLikeNone(parameters) ? 0 : addToExternrefTable0(parameters));
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Scroll through all points
   *
   * # Arguments
   * * `collection` - Collection name
   * * `options` - Scroll options: `{ limit?: number, offset?: number, with_payload?: boolean, with_vector?: boolean }`
   *
   * # Returns
   * Scroll result with points and next offset
   * @param {string} collection
   * @param {any | null} [options]
   * @returns {any}
   */
  scroll(collection, options) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_scroll(this.__wbg_ptr, ptr0, len0, isLikeNone(options) ? 0 : addToExternrefTable0(options));
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Search for nearest neighbors
   *
   * # Arguments
   * * `collection` - Collection name
   * * `vector` - Query vector
   * * `limit` - Maximum number of results
   * * `options` - Optional search options: `{ with_payload?: boolean, with_vector?: boolean, score_threshold?: number }`
   *
   * # Returns
   * Array of search results: `[{ id: number, score: number, payload?: object, vector?: number[] }, ...]`
   * @param {string} collection
   * @param {Float32Array} vector
   * @param {number} limit
   * @param {any | null} [options]
   * @returns {any}
   */
  search(collection, vector, limit, options) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(vector, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_search(this.__wbg_ptr, ptr0, len0, ptr1, len1, limit, isLikeNone(options) ? 0 : addToExternrefTable0(options));
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Traverse the graph from a starting point
   *
   * # Arguments
   * * `collection` - Collection name
   * * `start_id` - Starting point ID
   * * `max_depth` - Maximum traversal depth
   * * `relations` - Optional array of relation types to filter
   *
   * # Returns
   * Traversal result with visited nodes and edges
   * @param {string} collection
   * @param {bigint} start_id
   * @param {number} max_depth
   * @param {string[] | null} [relations]
   * @returns {any}
   */
  traverse(collection, start_id, max_depth, relations) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = isLikeNone(relations) ? 0 : passArrayJsValueToWasm0(relations, wasm.__wbindgen_malloc);
    var len1 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_traverse(this.__wbg_ptr, ptr0, len0, start_id, max_depth, ptr1, len1);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
  /**
   * Upsert points into a collection
   *
   * # Arguments
   * * `collection` - Collection name
   * * `points` - Array of points: `[{ id: number, vector: number[], payload?: object }, ...]`
   *
   * # Returns
   * Upsert result with operation status
   * @param {string} collection
   * @param {any} points_js
   * @returns {any}
   */
  upsert(collection, points_js) {
    const ptr0 = passStringToWasm0(collection, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.latticedb_upsert(this.__wbg_ptr, ptr0, len0, points_js);
    if (ret[2]) {
      throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
  }
};
if (Symbol.dispose) LatticeDB.prototype[Symbol.dispose] = LatticeDB.prototype.free;
function __wbg_get_imports() {
  const import0 = {
    __proto__: null,
    __wbg_Error_8c4e43fe74559d73: function(arg0, arg1) {
      const ret = Error(getStringFromWasm0(arg0, arg1));
      return ret;
    },
    __wbg_Number_04624de7d0e8332d: function(arg0) {
      const ret = Number(arg0);
      return ret;
    },
    __wbg_String_8f0eb39a4a4c2f66: function(arg0, arg1) {
      const ret = String(arg1);
      const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
      const len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
      getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    },
    __wbg___wbindgen_bigint_get_as_i64_8fcf4ce7f1ca72a2: function(arg0, arg1) {
      const v = arg1;
      const ret = typeof v === "bigint" ? v : void 0;
      getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
      getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    },
    __wbg___wbindgen_boolean_get_bbbb1c18aa2f5e25: function(arg0) {
      const v = arg0;
      const ret = typeof v === "boolean" ? v : void 0;
      return isLikeNone(ret) ? 16777215 : ret ? 1 : 0;
    },
    __wbg___wbindgen_debug_string_0bc8482c6e3508ae: function(arg0, arg1) {
      const ret = debugString(arg1);
      const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
      const len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
      getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    },
    __wbg___wbindgen_in_47fa6863be6f2f25: function(arg0, arg1) {
      const ret = arg0 in arg1;
      return ret;
    },
    __wbg___wbindgen_is_bigint_31b12575b56f32fc: function(arg0) {
      const ret = typeof arg0 === "bigint";
      return ret;
    },
    __wbg___wbindgen_is_function_0095a73b8b156f76: function(arg0) {
      const ret = typeof arg0 === "function";
      return ret;
    },
    __wbg___wbindgen_is_object_5ae8e5880f2c1fbd: function(arg0) {
      const val = arg0;
      const ret = typeof val === "object" && val !== null;
      return ret;
    },
    __wbg___wbindgen_is_undefined_9e4d92534c42d778: function(arg0) {
      const ret = arg0 === void 0;
      return ret;
    },
    __wbg___wbindgen_jsval_eq_11888390b0186270: function(arg0, arg1) {
      const ret = arg0 === arg1;
      return ret;
    },
    __wbg___wbindgen_jsval_loose_eq_9dd77d8cd6671811: function(arg0, arg1) {
      const ret = arg0 == arg1;
      return ret;
    },
    __wbg___wbindgen_number_get_8ff4255516ccad3e: function(arg0, arg1) {
      const obj = arg1;
      const ret = typeof obj === "number" ? obj : void 0;
      getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
      getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    },
    __wbg___wbindgen_string_get_72fb696202c56729: function(arg0, arg1) {
      const obj = arg1;
      const ret = typeof obj === "string" ? obj : void 0;
      var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
      var len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
      getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    },
    __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
      throw new Error(getStringFromWasm0(arg0, arg1));
    },
    __wbg_call_389efe28435a9388: function() {
      return handleError(function(arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
      }, arguments);
    },
    __wbg_done_57b39ecd9addfe81: function(arg0) {
      const ret = arg0.done;
      return ret;
    },
    __wbg_entries_58c7934c745daac7: function(arg0) {
      const ret = Object.entries(arg0);
      return ret;
    },
    __wbg_error_7534b8e9a36f1ab4: function(arg0, arg1) {
      let deferred0_0;
      let deferred0_1;
      try {
        deferred0_0 = arg0;
        deferred0_1 = arg1;
        console.error(getStringFromWasm0(arg0, arg1));
      } finally {
        wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
      }
    },
    __wbg_get_9b94d73e6221f75c: function(arg0, arg1) {
      const ret = arg0[arg1 >>> 0];
      return ret;
    },
    __wbg_get_b3ed3ad4be2bc8ac: function() {
      return handleError(function(arg0, arg1) {
        const ret = Reflect.get(arg0, arg1);
        return ret;
      }, arguments);
    },
    __wbg_get_with_ref_key_1dc361bd10053bfe: function(arg0, arg1) {
      const ret = arg0[arg1];
      return ret;
    },
    __wbg_instanceof_ArrayBuffer_c367199e2fa2aa04: function(arg0) {
      let result;
      try {
        result = arg0 instanceof ArrayBuffer;
      } catch (_) {
        result = false;
      }
      const ret = result;
      return ret;
    },
    __wbg_instanceof_Map_53af74335dec57f4: function(arg0) {
      let result;
      try {
        result = arg0 instanceof Map;
      } catch (_) {
        result = false;
      }
      const ret = result;
      return ret;
    },
    __wbg_instanceof_Uint8Array_9b9075935c74707c: function(arg0) {
      let result;
      try {
        result = arg0 instanceof Uint8Array;
      } catch (_) {
        result = false;
      }
      const ret = result;
      return ret;
    },
    __wbg_isArray_d314bb98fcf08331: function(arg0) {
      const ret = Array.isArray(arg0);
      return ret;
    },
    __wbg_isSafeInteger_bfbc7332a9768d2a: function(arg0) {
      const ret = Number.isSafeInteger(arg0);
      return ret;
    },
    __wbg_iterator_6ff6560ca1568e55: function() {
      const ret = Symbol.iterator;
      return ret;
    },
    __wbg_length_32ed9a279acd054c: function(arg0) {
      const ret = arg0.length;
      return ret;
    },
    __wbg_length_35a7bace40f36eac: function(arg0) {
      const ret = arg0.length;
      return ret;
    },
    __wbg_new_8a6f238a6ece86ea: function() {
      const ret = new Error();
      return ret;
    },
    __wbg_new_dd2b680c8bf6ae29: function(arg0) {
      const ret = new Uint8Array(arg0);
      return ret;
    },
    __wbg_new_no_args_1c7c842f08d00ebb: function(arg0, arg1) {
      const ret = new Function(getStringFromWasm0(arg0, arg1));
      return ret;
    },
    __wbg_next_3482f54c49e8af19: function() {
      return handleError(function(arg0) {
        const ret = arg0.next();
        return ret;
      }, arguments);
    },
    __wbg_next_418f80d8f5303233: function(arg0) {
      const ret = arg0.next;
      return ret;
    },
    __wbg_now_0dc4920a47cf7280: function(arg0) {
      const ret = arg0.now();
      return ret;
    },
    __wbg_parse_708461a1feddfb38: function() {
      return handleError(function(arg0, arg1) {
        const ret = JSON.parse(getStringFromWasm0(arg0, arg1));
        return ret;
      }, arguments);
    },
    __wbg_performance_6adc3b899e448a23: function(arg0) {
      const ret = arg0.performance;
      return ret;
    },
    __wbg_prototypesetcall_bdcdcc5842e4d77d: function(arg0, arg1, arg2) {
      Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
    },
    __wbg_stack_0ed75d68575b0f3c: function(arg0, arg1) {
      const ret = arg1.stack;
      const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
      const len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
      getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    },
    __wbg_static_accessor_GLOBAL_12837167ad935116: function() {
      const ret = typeof global === "undefined" ? null : global;
      return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    },
    __wbg_static_accessor_GLOBAL_THIS_e628e89ab3b1c95f: function() {
      const ret = typeof globalThis === "undefined" ? null : globalThis;
      return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    },
    __wbg_static_accessor_SELF_a621d3dfbb60d0ce: function() {
      const ret = typeof self === "undefined" ? null : self;
      return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    },
    __wbg_static_accessor_WINDOW_f8727f0cf888e0bd: function() {
      const ret = typeof window === "undefined" ? null : window;
      return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    },
    __wbg_value_0546255b415e96c1: function(arg0) {
      const ret = arg0.value;
      return ret;
    },
    __wbindgen_cast_0000000000000001: function(arg0) {
      const ret = arg0;
      return ret;
    },
    __wbindgen_cast_0000000000000002: function(arg0, arg1) {
      const ret = getStringFromWasm0(arg0, arg1);
      return ret;
    },
    __wbindgen_cast_0000000000000003: function(arg0) {
      const ret = BigInt.asUintN(64, arg0);
      return ret;
    },
    __wbindgen_init_externref_table: function() {
      const table = wasm.__wbindgen_externrefs;
      const offset = table.grow(4);
      table.set(0, void 0);
      table.set(offset + 0, void 0);
      table.set(offset + 1, null);
      table.set(offset + 2, true);
      table.set(offset + 3, false);
    },
    __wbindgen_object_is_undefined: function(arg0) {
      const ret = arg0 === void 0;
      return ret;
    }
  };
  return {
    __proto__: null,
    "./lattice_server_bg.js": import0
  };
}
var LatticeDBFinalization = typeof FinalizationRegistry === "undefined" ? { register: () => {
}, unregister: () => {
} } : new FinalizationRegistry((ptr) => wasm.__wbg_latticedb_free(ptr >>> 0, 1));
function addToExternrefTable0(obj) {
  const idx = wasm.__externref_table_alloc();
  wasm.__wbindgen_externrefs.set(idx, obj);
  return idx;
}
function debugString(val) {
  const type = typeof val;
  if (type == "number" || type == "boolean" || val == null) {
    return `${val}`;
  }
  if (type == "string") {
    return `"${val}"`;
  }
  if (type == "symbol") {
    const description = val.description;
    if (description == null) {
      return "Symbol";
    } else {
      return `Symbol(${description})`;
    }
  }
  if (type == "function") {
    const name = val.name;
    if (typeof name == "string" && name.length > 0) {
      return `Function(${name})`;
    } else {
      return "Function";
    }
  }
  if (Array.isArray(val)) {
    const length = val.length;
    let debug = "[";
    if (length > 0) {
      debug += debugString(val[0]);
    }
    for (let i = 1; i < length; i++) {
      debug += ", " + debugString(val[i]);
    }
    debug += "]";
    return debug;
  }
  const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
  let className;
  if (builtInMatches && builtInMatches.length > 1) {
    className = builtInMatches[1];
  } else {
    return toString.call(val);
  }
  if (className == "Object") {
    try {
      return "Object(" + JSON.stringify(val) + ")";
    } catch (_) {
      return "Object";
    }
  }
  if (val instanceof Error) {
    return `${val.name}: ${val.message}
${val.stack}`;
  }
  return className;
}
function getArrayU8FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}
var cachedBigUint64ArrayMemory0 = null;
function getBigUint64ArrayMemory0() {
  if (cachedBigUint64ArrayMemory0 === null || cachedBigUint64ArrayMemory0.byteLength === 0) {
    cachedBigUint64ArrayMemory0 = new BigUint64Array(wasm.memory.buffer);
  }
  return cachedBigUint64ArrayMemory0;
}
var cachedDataViewMemory0 = null;
function getDataViewMemory0() {
  if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || cachedDataViewMemory0.buffer.detached === void 0 && cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
    cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
  }
  return cachedDataViewMemory0;
}
var cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
  if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
    cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
  }
  return cachedFloat32ArrayMemory0;
}
function getStringFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return decodeText(ptr, len);
}
var cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
  if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
    cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
  }
  return cachedUint8ArrayMemory0;
}
function handleError(f, args) {
  try {
    return f.apply(this, args);
  } catch (e) {
    const idx = addToExternrefTable0(e);
    wasm.__wbindgen_exn_store(idx);
  }
}
function isLikeNone(x) {
  return x === void 0 || x === null;
}
function passArray64ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 8, 8) >>> 0;
  getBigUint64ArrayMemory0().set(arg, ptr / 8);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}
function passArrayF32ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 4, 4) >>> 0;
  getFloat32ArrayMemory0().set(arg, ptr / 4);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}
function passArrayJsValueToWasm0(array, malloc) {
  const ptr = malloc(array.length * 4, 4) >>> 0;
  for (let i = 0; i < array.length; i++) {
    const add = addToExternrefTable0(array[i]);
    getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
  }
  WASM_VECTOR_LEN = array.length;
  return ptr;
}
function passStringToWasm0(arg, malloc, realloc) {
  if (realloc === void 0) {
    const buf = cachedTextEncoder.encode(arg);
    const ptr2 = malloc(buf.length, 1) >>> 0;
    getUint8ArrayMemory0().subarray(ptr2, ptr2 + buf.length).set(buf);
    WASM_VECTOR_LEN = buf.length;
    return ptr2;
  }
  let len = arg.length;
  let ptr = malloc(len, 1) >>> 0;
  const mem = getUint8ArrayMemory0();
  let offset = 0;
  for (; offset < len; offset++) {
    const code = arg.charCodeAt(offset);
    if (code > 127) break;
    mem[ptr + offset] = code;
  }
  if (offset !== len) {
    if (offset !== 0) {
      arg = arg.slice(offset);
    }
    ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
    const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
    const ret = cachedTextEncoder.encodeInto(arg, view);
    offset += ret.written;
    ptr = realloc(ptr, len, offset, 1) >>> 0;
  }
  WASM_VECTOR_LEN = offset;
  return ptr;
}
function takeFromExternrefTable0(idx) {
  const value = wasm.__wbindgen_externrefs.get(idx);
  wasm.__externref_table_dealloc(idx);
  return value;
}
var cachedTextDecoder = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
var MAX_SAFARI_DECODE_BYTES = 2146435072;
var numBytesDecoded = 0;
function decodeText(ptr, len) {
  numBytesDecoded += len;
  if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
    cachedTextDecoder = new TextDecoder("utf-8", { ignoreBOM: true, fatal: true });
    cachedTextDecoder.decode();
    numBytesDecoded = len;
  }
  return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}
var cachedTextEncoder = new TextEncoder();
if (!("encodeInto" in cachedTextEncoder)) {
  cachedTextEncoder.encodeInto = function(arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
      read: arg.length,
      written: buf.length
    };
  };
}
var WASM_VECTOR_LEN = 0;
var wasmModule;
var wasm;
function __wbg_finalize_init(instance, module) {
  wasm = instance.exports;
  wasmModule = module;
  cachedBigUint64ArrayMemory0 = null;
  cachedDataViewMemory0 = null;
  cachedFloat32ArrayMemory0 = null;
  cachedUint8ArrayMemory0 = null;
  wasm.__wbindgen_start();
  return wasm;
}
async function __wbg_load(module, imports) {
  if (typeof Response === "function" && module instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming === "function") {
      try {
        return await WebAssembly.instantiateStreaming(module, imports);
      } catch (e) {
        const validResponse = module.ok && expectedResponseType(module.type);
        if (validResponse && module.headers.get("Content-Type") !== "application/wasm") {
          console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);
        } else {
          throw e;
        }
      }
    }
    const bytes = await module.arrayBuffer();
    return await WebAssembly.instantiate(bytes, imports);
  } else {
    const instance = await WebAssembly.instantiate(module, imports);
    if (instance instanceof WebAssembly.Instance) {
      return { instance, module };
    } else {
      return instance;
    }
  }
  function expectedResponseType(type) {
    switch (type) {
      case "basic":
      case "cors":
      case "default":
        return true;
    }
    return false;
  }
}
function initSync(module) {
  if (wasm !== void 0) return wasm;
  if (module !== void 0) {
    if (Object.getPrototypeOf(module) === Object.prototype) {
      ({ module } = module);
    } else {
      console.warn("using deprecated parameters for `initSync()`; pass a single object instead");
    }
  }
  const imports = __wbg_get_imports();
  if (!(module instanceof WebAssembly.Module)) {
    module = new WebAssembly.Module(module);
  }
  const instance = new WebAssembly.Instance(module, imports);
  return __wbg_finalize_init(instance, module);
}
async function __wbg_init(module_or_path) {
  if (wasm !== void 0) return wasm;
  if (module_or_path !== void 0) {
    if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
      ({ module_or_path } = module_or_path);
    } else {
      console.warn("using deprecated parameters for the initialization function; pass a single object instead");
    }
  }
  if (module_or_path === void 0) {
    module_or_path = new URL("lattice_server_bg.wasm", import.meta.url);
  }
  const imports = __wbg_get_imports();
  if (typeof module_or_path === "string" || typeof Request === "function" && module_or_path instanceof Request || typeof URL === "function" && module_or_path instanceof URL) {
    module_or_path = fetch(module_or_path);
  }
  const { instance, module } = await __wbg_load(await module_or_path, imports);
  return __wbg_finalize_init(instance, module);
}
export {
  LatticeDB,
  __wbg_init as default,
  initSync
};
//# sourceMappingURL=lattice_server-Q6DPRQB4.js.map
